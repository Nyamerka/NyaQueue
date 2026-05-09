package broker

import (
	"encoding/binary"
	"errors"
	"sync"
	"time"
)

// Wire layout v1: [version:1][priority:1][timestamp:8][keyLen:4][key:keyLen][value:...]
// Version 0 (legacy): [priority:1][timestamp:8][keyLen:4][key:keyLen][value:...]
//
// On read, version is detected by checking if the first byte is a known version tag.
// Version 1+ use byte 0 as version; version 0 (legacy) has priority in byte 0.
const (
	currentVersion     = 1
	versionFieldSize   = 1
	priorityFieldSize  = 1
	timestampFieldSize = 8
	keyLenFieldSize    = 4

	// v1 offsets
	v1PriorityOffset = versionFieldSize                                                            // 1
	v1TimestampOff   = versionFieldSize + priorityFieldSize                                        // 2
	v1KeyLenOff      = versionFieldSize + priorityFieldSize + timestampFieldSize                   // 10
	v1KeyOff         = versionFieldSize + priorityFieldSize + timestampFieldSize + keyLenFieldSize // 14

	// v0 (legacy) offsets — version field is absent
	timestampOffset = priorityFieldSize                      // 1
	keyLenOffset    = priorityFieldSize + timestampFieldSize // 9 (== HeaderSize)
	keyOffset       = keyLenOffset + keyLenFieldSize         // 13

	// HeaderSize covers the fixed-width prefix (v0 format).
	HeaderSize = keyLenOffset

	// metadataSize for v0 format
	metadataSize = keyOffset

	// v1MetadataSize includes version byte
	v1MetadataSize = v1KeyOff
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
	Version     uint8
	Priority    uint8
	Timestamp   int64 // unix nanos — client enqueue time
	ProduceTime int64 // unix nanos — time when broker receives the message
	AppendTime  int64 // unix nanos — time when written to WAL
}

type Message struct {
	Header       MessageHeader
	Key          []byte
	Value        []byte
	BatchDecoded bool `json:"-"` // true when Value was already decompressed by batch read
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
	return v1MetadataSize + len(m.Key) + len(m.Value)
}

func (m *Message) marshalInto(buf []byte) {
	keyLen := len(m.Key)
	buf[0] = currentVersion
	buf[v1PriorityOffset] = m.Header.Priority
	binary.BigEndian.PutUint64(buf[v1TimestampOff:v1KeyLenOff], uint64(m.Header.Timestamp))
	binary.BigEndian.PutUint32(buf[v1KeyLenOff:v1KeyOff], uint32(keyLen))
	copy(buf[v1KeyOff:v1KeyOff+keyLen], m.Key)
	copy(buf[v1KeyOff+keyLen:], m.Value)
}

func UnmarshalMessage(data []byte) (*Message, error) {
	if len(data) < metadataSize {
		return nil, errors.New("message data too short")
	}

	version := data[0]

	if version == currentVersion {
		if len(data) < v1MetadataSize {
			return nil, errors.New("v1 message data too short")
		}
		priority := data[v1PriorityOffset]
		timestamp := int64(binary.BigEndian.Uint64(data[v1TimestampOff:v1KeyLenOff]))
		keyLen := int(binary.BigEndian.Uint32(data[v1KeyLenOff:v1KeyOff]))

		if len(data) < v1MetadataSize+keyLen {
			return nil, errors.New("v1 message data truncated")
		}

		key := make([]byte, keyLen)
		copy(key, data[v1KeyOff:v1KeyOff+keyLen])

		value := make([]byte, len(data)-(v1KeyOff+keyLen))
		copy(value, data[v1KeyOff+keyLen:])

		return &Message{
			Header: MessageHeader{
				Version:   version,
				Priority:  priority,
				Timestamp: timestamp,
			},
			Key:   key,
			Value: value,
		}, nil
	}

	// v0 (legacy) format: no version byte, first byte is priority.
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

// UnmarshalHeader reads the fixed header for fast PriorityIndex rebuild.
// Handles both v0 and v1 formats.
func UnmarshalHeader(data []byte) (MessageHeader, error) {
	if len(data) < HeaderSize {
		return MessageHeader{}, errors.New("data too short for header")
	}

	version := data[0]
	if version == currentVersion {
		if len(data) < v1TimestampOff+timestampFieldSize {
			return MessageHeader{}, errors.New("v1 data too short for header")
		}
		return MessageHeader{
			Version:   version,
			Priority:  data[v1PriorityOffset],
			Timestamp: int64(binary.BigEndian.Uint64(data[v1TimestampOff : v1TimestampOff+timestampFieldSize])),
		}, nil
	}

	return MessageHeader{
		Priority:  data[0],
		Timestamp: int64(binary.BigEndian.Uint64(data[timestampOffset:keyLenOffset])),
	}, nil
}
