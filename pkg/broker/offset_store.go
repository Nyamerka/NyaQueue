package broker

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/samber/oops"

	bbolt "go.etcd.io/bbolt"
)

var offsetsBucket = []byte("offsets")

// OffsetStore tracks consumer offsets. Hot-path reads and writes go through an
// in-memory cache (sync.Map). A background goroutine periodically dumps dirty
// entries to BoltDB for durability.
type OffsetStore struct {
	db    *bbolt.DB
	cache sync.Map // string -> int64

	dumpInterval time.Duration
	stopCh       chan struct{}
	stopped      chan struct{}
}

// NewOffsetStore opens the offset database. When dumpInterval > 0 the store
// runs a background goroutine that flushes the in-memory cache to BoltDB at
// that cadence, removing the fsync-per-commit overhead from the hot path.
func NewOffsetStore(dataDir string, dumpInterval ...time.Duration) (*OffsetStore, error) {
	path := filepath.Join(dataDir, "offsets.db")
	db, err := bbolt.Open(path, 0600, nil)
	if err != nil {
		return nil, oops.Wrapf(err, "open offset store")
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(offsetsBucket)
		return err
	})
	if err != nil {
		db.Close()
		return nil, err
	}

	s := &OffsetStore{
		db:      db,
		stopCh:  make(chan struct{}),
		stopped: make(chan struct{}),
	}

	// Warm cache from BoltDB.
	_ = db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		return b.ForEach(func(k, v []byte) error {
			s.cache.Store(string(k), int64(binary.BigEndian.Uint64(v)))
			return nil
		})
	})

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
// BoltDB by the background dump goroutine (or synchronously when no dump
// interval is configured).
func (s *OffsetStore) Commit(group, topic string, partition int, offset int64) error {
	key := s.bucketKey(topic, partition, group)
	s.cache.Store(key, offset)

	if s.dumpInterval > 0 {
		return nil
	}

	return s.writeOne(key, offset)
}

// Load returns the last committed offset for a consumer group.
func (s *OffsetStore) Load(group, topic string, partition int) (int64, error) {
	key := s.bucketKey(topic, partition, group)
	if v, ok := s.cache.Load(key); ok {
		return v.(int64), nil
	}
	return 0, oops.Errorf("offset not found for %s/%d/%s", topic, partition, group)
}

// CommitFloor returns the minimum committed offset across all consumer groups
// for a given topic/partition. Used for WAL truncation and PriorityIndex rebuild.
func (s *OffsetStore) CommitFloor(topic string, partition int) (int64, error) {
	prefix := fmt.Sprintf("%s/%d/", topic, partition)
	minOffset := int64(math.MaxInt64)
	found := false

	s.cache.Range(func(k, v any) bool {
		key := k.(string)
		if strings.HasPrefix(key, prefix) {
			off := v.(int64)
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
	s.cache.Range(func(k, _ any) bool {
		if strings.HasPrefix(k.(string), prefix) {
			keys = append(keys, k.(string))
		}
		return true
	})
	for _, k := range keys {
		s.cache.Delete(k)
	}

	_ = s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		for _, k := range keys {
			_ = b.Delete([]byte(k))
		}
		return nil
	})
}

func (s *OffsetStore) writeOne(key string, offset int64) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		val := make([]byte, 8)
		binary.BigEndian.PutUint64(val, uint64(offset))
		return b.Put([]byte(key), val)
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
	snapshot := make(map[string]int64)
	s.cache.Range(func(k, v any) bool {
		snapshot[k.(string)] = v.(int64)
		return true
	})
	if len(snapshot) == 0 {
		return
	}

	_ = s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		for key, offset := range snapshot {
			val := make([]byte, 8)
			binary.BigEndian.PutUint64(val, uint64(offset))
			_ = b.Put([]byte(key), val)
		}
		return nil
	})
}

func (s *OffsetStore) Close() error {
	if s.dumpInterval > 0 {
		close(s.stopCh)
		<-s.stopped
	}
	return s.db.Close()
}
