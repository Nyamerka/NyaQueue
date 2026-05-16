package broker

import (
	"encoding/binary"
	"sync"
	"time"

	"github.com/samber/oops"
)

// Wire layout: [version:1][priority:1][timestamp:8][keyLen:4][key:keyLen][value:...]
const (
	currentVersion     = 1
	versionFieldSize   = 1
	priorityFieldSize  = 1
	timestampFieldSize = 8
	keyLenFieldSize    = 4

	priorityOffset = versionFieldSize                                                            // 1
	timestampOff   = versionFieldSize + priorityFieldSize                                        // 2
	keyLenOff      = versionFieldSize + priorityFieldSize + timestampFieldSize                   // 10
	keyOff         = versionFieldSize + priorityFieldSize + timestampFieldSize + keyLenFieldSize // 14

	// HeaderSize is the fixed-width prefix before key/value data.
	HeaderSize = keyLenOff

	// MetadataSize includes version, priority, timestamp, and key length fields.
	MetadataSize = keyOff
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
	return MetadataSize + len(m.Key) + len(m.Value)
}

func (m *Message) marshalInto(buf []byte) {
	keyLen := len(m.Key)
	buf[0] = currentVersion
	buf[priorityOffset] = m.Header.Priority
	binary.BigEndian.PutUint64(buf[timestampOff:keyLenOff], uint64(m.Header.Timestamp))
	binary.BigEndian.PutUint32(buf[keyLenOff:keyOff], uint32(keyLen))
	copy(buf[keyOff:keyOff+keyLen], m.Key)
	copy(buf[keyOff+keyLen:], m.Value)
}

func UnmarshalMessage(data []byte) (*Message, error) {
	if len(data) < MetadataSize {
		return nil, oops.Errorf("message data too short: got %d bytes, need at least %d", len(data), MetadataSize)
	}

	version := data[0]
	if version != currentVersion {
		return nil, oops.Errorf("unsupported message version %d (expected %d)", version, currentVersion)
	}

	priority := data[priorityOffset]
	timestamp := int64(binary.BigEndian.Uint64(data[timestampOff:keyLenOff]))
	keyLen := int(binary.BigEndian.Uint32(data[keyLenOff:keyOff]))

	if len(data) < MetadataSize+keyLen {
		return nil, oops.Errorf("message truncated: got %d bytes, need %d (keyLen=%d)", len(data), MetadataSize+keyLen, keyLen)
	}

	key := make([]byte, keyLen)
	copy(key, data[keyOff:keyOff+keyLen])

	value := make([]byte, len(data)-(keyOff+keyLen))
	copy(value, data[keyOff+keyLen:])

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
func UnmarshalHeader(data []byte) (MessageHeader, error) {
	if len(data) < HeaderSize {
		return MessageHeader{}, oops.Errorf("data too short for header: got %d bytes, need at least %d", len(data), HeaderSize)
	}

	version := data[0]
	if version != currentVersion {
		return MessageHeader{}, oops.Errorf("unsupported message version %d", version)
	}

	return MessageHeader{
		Priority:  data[priorityOffset],
		Timestamp: int64(binary.BigEndian.Uint64(data[timestampOff : timestampOff+timestampFieldSize])),
	}, nil
}
