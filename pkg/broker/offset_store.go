package broker

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"sync"
	"time"

	"github.com/samber/oops"

	bbolt "go.etcd.io/bbolt"
)

var offsetsBucket = []byte("offsets")

type OffsetStore struct {
	db *bbolt.DB

	batchMu       sync.Mutex
	pendingWrites map[string]int64 // bucketKey -> offset
	batchInterval time.Duration
	stopCh        chan struct{}
	stopped       chan struct{}
}

// NewOffsetStore opens the offset database. When batchInterval > 0, commits
// are accumulated in memory and flushed to disk periodically, trading a small
// durability window for much higher throughput.
func NewOffsetStore(dataDir string, batchInterval ...time.Duration) (*OffsetStore, error) {
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
		db:            db,
		pendingWrites: make(map[string]int64),
		stopCh:        make(chan struct{}),
		stopped:       make(chan struct{}),
	}
	if len(batchInterval) > 0 && batchInterval[0] > 0 {
		s.batchInterval = batchInterval[0]
		go s.flushLoop()
	}
	return s, nil
}

func (s *OffsetStore) bucketKey(topic string, partition int, group string) []byte {
	return []byte(fmt.Sprintf("%s/%d/%s", topic, partition, group))
}

func (s *OffsetStore) Commit(group, topic string, partition int, offset int64) error {
	key := string(s.bucketKey(topic, partition, group))

	if s.batchInterval > 0 {
		s.batchMu.Lock()
		s.pendingWrites[key] = offset
		s.batchMu.Unlock()
		return nil
	}

	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		val := make([]byte, 8)
		binary.BigEndian.PutUint64(val, uint64(offset))
		return b.Put([]byte(key), val)
	})
}

func (s *OffsetStore) flushLoop() {
	defer close(s.stopped)
	ticker := time.NewTicker(s.batchInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.flush()
		case <-s.stopCh:
			s.flush()
			return
		}
	}
}

func (s *OffsetStore) flush() {
	s.batchMu.Lock()
	batch := s.pendingWrites
	s.pendingWrites = make(map[string]int64, len(batch))
	s.batchMu.Unlock()

	if len(batch) == 0 {
		return
	}

	_ = s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		val := make([]byte, 8)
		for key, offset := range batch {
			binary.BigEndian.PutUint64(val, uint64(offset))
			_ = b.Put([]byte(key), val)
		}
		return nil
	})
}

func (s *OffsetStore) Load(group, topic string, partition int) (int64, error) {
	key := string(s.bucketKey(topic, partition, group))

	if s.batchInterval > 0 {
		s.batchMu.Lock()
		if off, ok := s.pendingWrites[key]; ok {
			s.batchMu.Unlock()
			return off, nil
		}
		s.batchMu.Unlock()
	}

	var offset int64
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		val := b.Get([]byte(key))
		if val == nil {
			return oops.Errorf("offset not found for %s/%s/%d", group, topic, partition)
		}
		offset = int64(binary.BigEndian.Uint64(val))
		return nil
	})
	return offset, err
}

// CommitFloor returns the minimum committed offset across all consumer groups
// for a given topic/partition. Used for WAL truncation and PriorityIndex rebuild.
func (s *OffsetStore) CommitFloor(topic string, partition int) (int64, error) {
	prefix := []byte(fmt.Sprintf("%s/%d/", topic, partition))
	minOffset := int64(math.MaxInt64)
	found := false

	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		c := b.Cursor()

		for k, v := c.Seek(prefix); k != nil && len(k) >= len(prefix) && string(k[:len(prefix)]) == string(prefix); k, v = c.Next() {
			off := int64(binary.BigEndian.Uint64(v))
			if off < minOffset {
				minOffset = off
			}
			found = true
		}
		return nil
	})

	if err != nil {
		return 0, err
	}
	if !found {
		return 0, nil
	}
	return minOffset, nil
}

func (s *OffsetStore) Close() error {
	if s.batchInterval > 0 {
		close(s.stopCh)
		<-s.stopped
	}
	return s.db.Close()
}
