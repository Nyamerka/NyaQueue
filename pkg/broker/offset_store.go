package broker

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strings"
	"time"

	"github.com/dgraph-io/badger/v4"
	"github.com/puzpuzpuz/xsync/v3"
	"github.com/samber/oops"
)

var offsetsPrefix = []byte("off/")

// OffsetStore tracks consumer offsets with session TTL support. Hot-path reads
// and writes go through an in-memory cache (xsync.MapOf — cache-line hash
// table, lock-free reads). A background goroutine periodically flushes dirty
// entries to BadgerDB via WriteBatch for durability.
//
// Trade-off: when dumpInterval > 0, offsets committed between the last
// successful flush and a crash are lost. On restart consumers will re-receive
// messages from the last persisted offset (at-least-once semantics). This is a
// deliberate compromise to keep fsync off the hot path; set dumpInterval to 0
// (or use SyncEveryWrite) for stronger guarantees at the cost of throughput.
type OffsetStore struct {
	db         *badger.DB
	cache      *xsync.MapOf[string, int64]
	lastCommit *xsync.MapOf[string, time.Time]

	dumpInterval time.Duration
	stopCh       chan struct{}
	stopped      chan struct{}
}

func badgerKey(cacheKey string) []byte {
	return append(offsetsPrefix, []byte(cacheKey)...)
}

func badgerOpts(dir string) badger.Options {
	return badger.DefaultOptions(dir).
		WithLogger(nil).
		WithNumVersionsToKeep(1).
		WithCompactL0OnClose(true).
		WithValueLogFileSize(16 << 20)
}

// NewOffsetStore opens the offset database backed by BadgerDB. When
// dumpInterval > 0 the store runs a background goroutine that flushes the
// in-memory cache to BadgerDB at that cadence via WriteBatch, removing the
// fsync-per-commit overhead from the hot path.
func NewOffsetStore(dataDir string, dumpInterval ...time.Duration) (*OffsetStore, error) {
	dir := filepath.Join(dataDir, "offsets.badger")
	db, err := badger.Open(badgerOpts(dir))
	if err != nil {
		return nil, oops.Wrapf(err, "open offset store")
	}

	s := &OffsetStore{
		db:         db,
		cache:      xsync.NewMapOf[string, int64](),
		lastCommit: xsync.NewMapOf[string, time.Time](),
		stopCh:     make(chan struct{}),
		stopped:    make(chan struct{}),
	}

	err = db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.Prefix = offsetsPrefix
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			key := string(item.Key()[len(offsetsPrefix):])
			if err := item.Value(func(val []byte) error {
				if len(val) == 8 {
					s.cache.Store(key, int64(binary.BigEndian.Uint64(val)))
				}
				return nil
			}); err != nil {
				return oops.Wrapf(err, "read offset value for key %q", key)
			}
		}
		return nil
	})
	if err != nil {
		db.Close()
		return nil, oops.Wrapf(err, "warm offset cache")
	}

	if len(dumpInterval) > 0 && dumpInterval[0] > 0 {
		s.dumpInterval = dumpInterval[0]
		go s.dumpLoop()
	}

	return s, nil
}

func (s *OffsetStore) bucketKey(topic string, partition int, group string) string {
	return fmt.Sprintf("%s/%d/%s", topic, partition, group)
}

// Commit stores the offset in the in-memory cache. The value is persisted to
// BadgerDB by the background dump goroutine (or synchronously when no dump
// interval is configured).
func (s *OffsetStore) Commit(group, topic string, partition int, offset int64) error {
	key := s.bucketKey(topic, partition, group)
	s.cache.Store(key, offset)
	s.lastCommit.Store(key, time.Now())

	if s.dumpInterval > 0 {
		return nil
	}

	return s.writeOne(key, offset)
}

// Load returns the last committed offset for a consumer group.
func (s *OffsetStore) Load(group, topic string, partition int) (int64, error) {
	key := s.bucketKey(topic, partition, group)
	if v, ok := s.cache.Load(key); ok {
		return v, nil
	}
	return 0, oops.Errorf("offset not found for %s/%d/%s", topic, partition, group)
}

// CommitFloor returns the minimum committed offset across all consumer groups
// for a given topic/partition. Used for WAL truncation and PriorityIndex rebuild.
func (s *OffsetStore) CommitFloor(topic string, partition int) (int64, error) {
	prefix := fmt.Sprintf("%s/%d/", topic, partition)
	minOffset := int64(math.MaxInt64)
	found := false

	s.cache.Range(func(key string, off int64) bool {
		if strings.HasPrefix(key, prefix) {
			if off < minOffset {
				minOffset = off
			}
			found = true
		}
		return true
	})

	if !found {
		return 0, nil
	}
	return minOffset, nil
}

func (s *OffsetStore) DeleteTopic(topic string) {
	prefix := topic + "/"
	var keys []string
	s.cache.Range(func(key string, _ int64) bool {
		if strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
		return true
	})
	for _, k := range keys {
		s.cache.Delete(k)
	}

	wb := s.db.NewWriteBatch()
	for _, k := range keys {
		_ = wb.Delete(badgerKey(k))
	}
	if err := wb.Flush(); err != nil {
		log.Printf("offset store: delete topic %q from BadgerDB failed: %v", topic, err)
	}
}

func (s *OffsetStore) writeOne(key string, offset int64) error {
	val := make([]byte, 8)
	binary.BigEndian.PutUint64(val, uint64(offset))
	return s.db.Update(func(txn *badger.Txn) error {
		return txn.Set(badgerKey(key), val)
	})
}

func (s *OffsetStore) dumpLoop() {
	defer close(s.stopped)
	ticker := time.NewTicker(s.dumpInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.dump()
		case <-s.stopCh:
			s.dump()
			return
		}
	}
}

func (s *OffsetStore) dump() {
	wb := s.db.NewWriteBatch()
	count := 0

	s.cache.Range(func(key string, offset int64) bool {
		val := make([]byte, 8)
		binary.BigEndian.PutUint64(val, uint64(offset))
		if err := wb.Set(badgerKey(key), val); err != nil {
			log.Printf("offset store: WriteBatch.Set failed for key %q: %v", key, err)
		}
		count++
		return true
	})

	if count == 0 {
		wb.Cancel()
		return
	}

	if err := wb.Flush(); err != nil {
		log.Printf("offset store: flush to BadgerDB failed (%d entries): %v", count, err)
	}
}

// ExpireSessions removes consumer group offsets that haven't been committed
// within the given timeout, implementing ConsumerSessionTimeoutMs.
func (s *OffsetStore) ExpireSessions(timeout time.Duration) {
	cutoff := time.Now().Add(-timeout)
	var expired []string
	s.lastCommit.Range(func(key string, lastTime time.Time) bool {
		if lastTime.Before(cutoff) {
			expired = append(expired, key)
		}
		return true
	})
	for _, k := range expired {
		s.cache.Delete(k)
		s.lastCommit.Delete(k)
	}
}

func (s *OffsetStore) Close() error {
	if s.dumpInterval > 0 {
		close(s.stopCh)
		<-s.stopped
	}
	return s.db.Close()
}
