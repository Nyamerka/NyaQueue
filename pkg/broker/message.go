package broker

import (
	"encoding/binary"
	"errors"
	"sync"
	"time"
)

// Wire layout: [priority:1][timestamp:8][keyLen:4][key:keyLen][value:...]
const (
	priorityFieldSize  = 1
	timestampFieldSize = 8
	keyLenFieldSize    = 4

	timestampOffset = priorityFieldSize                  // 1
	keyLenOffset    = priorityFieldSize + timestampFieldSize // 9 (== HeaderSize)
	keyOffset       = keyLenOffset + keyLenFieldSize         // 13

	// HeaderSize covers the fixed-width prefix without the variable key length
	// field. UnmarshalHeader reads exactly these bytes to rebuild the
	// PriorityIndex without paying for a full key/value copy.
	HeaderSize = keyLenOffset

	// metadataSize is the full prefix including keyLen, i.e. the offset where
	// the actual key bytes start.
	metadataSize = keyOffset
)

// marshalPool holds reusable encode buffers as *[]byte to avoid the
// per-Put allocation that sync.Pool causes for non-pointer values (SA6002).
var marshalPool = sync.Pool{
	New: func() any {
		buf := make([]byte, 0, 1024)
		return &buf
	},
}

type MessageHeader struct {
	Priority  uint8
	Timestamp int64 // unix nanos
}

type Message struct {
	Header MessageHeader
	Key    []byte
	Value  []byte
}

func NewMessage(priority uint8, key, value []byte) *Message {
	return &Message{
		Header: MessageHeader{
			Priority:  priority,
			Timestamp: time.Now().UnixNano(),
		},
		Key:   key,
		Value: value,
	}
}

func (m *Message) Marshal() []byte {
	buf := make([]byte, m.encodedSize())
	m.marshalInto(buf)
	return buf
}

// MarshalPooled returns an encoded buffer drawn from a sync.Pool. Callers MUST
// pair every successful call with ReleaseMarshalBuf once they are done with
// the bytes (typically after the WAL has copied them).
func (m *Message) MarshalPooled() []byte {
	needed := m.encodedSize()

	bufPtr := marshalPool.Get().(*[]byte)
	buf := *bufPtr
	if cap(buf) < needed {
		buf = make([]byte, needed)
	} else {
		buf = buf[:needed]
	}

	m.marshalInto(buf)
	return buf
}

func ReleaseMarshalBuf(buf []byte) {
	if buf == nil {
		return
	}
	buf = buf[:0]
	marshalPool.Put(&buf)
}

func (m *Message) encodedSize() int {
	return metadataSize + len(m.Key) + len(m.Value)
}

func (m *Message) marshalInto(buf []byte) {
	keyLen := len(m.Key)
	buf[0] = m.Header.Priority
	binary.BigEndian.PutUint64(buf[timestampOffset:keyLenOffset], uint64(m.Header.Timestamp))
	binary.BigEndian.PutUint32(buf[keyLenOffset:keyOffset], uint32(keyLen))
	copy(buf[keyOffset:keyOffset+keyLen], m.Key)
	copy(buf[keyOffset+keyLen:], m.Value)
}

func UnmarshalMessage(data []byte) (*Message, error) {
	if len(data) < metadataSize {
		return nil, errors.New("message data too short")
	}

	priority := data[0]
	timestamp := int64(binary.BigEndian.Uint64(data[timestampOffset:keyLenOffset]))
	keyLen := int(binary.BigEndian.Uint32(data[keyLenOffset:keyOffset]))

	if len(data) < metadataSize+keyLen {
		return nil, errors.New("message data truncated")
	}

	key := make([]byte, keyLen)
	copy(key, data[keyOffset:keyOffset+keyLen])

	value := make([]byte, len(data)-(keyOffset+keyLen))
	copy(value, data[keyOffset+keyLen:])

	return &Message{
		Header: MessageHeader{
			Priority:  priority,
			Timestamp: timestamp,
		},
		Key:   key,
		Value: value,
	}, nil
}

// UnmarshalHeader reads only the fixed header (9 bytes) for fast PriorityIndex rebuild.
func UnmarshalHeader(data []byte) (MessageHeader, error) {
	if len(data) < HeaderSize {
		return MessageHeader{}, errors.New("data too short for header")
	}
	return MessageHeader{
		Priority:  data[0],
		Timestamp: int64(binary.BigEndian.Uint64(data[timestampOffset:keyLenOffset])),
	}, nil
}