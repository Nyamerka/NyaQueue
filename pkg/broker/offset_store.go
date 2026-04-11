package broker

import (
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"

	bbolt "go.etcd.io/bbolt"
)

var offsetsBucket = []byte("offsets")

type OffsetStore struct {
	db *bbolt.DB
}

func NewOffsetStore(dataDir string) (*OffsetStore, error) {
	path := filepath.Join(dataDir, "offsets.db")
	db, err := bbolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("open offset store: %w", err)
	}

	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(offsetsBucket)
		return err
	})
	if err != nil {
		db.Close()
		return nil, err
	}

	return &OffsetStore{db: db}, nil
}

func (s *OffsetStore) bucketKey(topic string, partition int, group string) []byte {
	return []byte(fmt.Sprintf("%s/%d/%s", topic, partition, group))
}

func (s *OffsetStore) Commit(group, topic string, partition int, offset int64) error {
	return s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		key := s.bucketKey(topic, partition, group)

		val := make([]byte, 8)
		binary.BigEndian.PutUint64(val, uint64(offset))

		return b.Put(key, val)
	})
}

func (s *OffsetStore) Load(group, topic string, partition int) (int64, error) {
	var offset int64
	err := s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket(offsetsBucket)
		key := s.bucketKey(topic, partition, group)
		val := b.Get(key)
		if val == nil {
			return fmt.Errorf("offset not found for %s/%s/%d", group, topic, partition)
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
	return s.db.Close()
}
