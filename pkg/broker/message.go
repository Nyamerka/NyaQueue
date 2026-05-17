package broker

import (
	"encoding/binary"
	"sync"
	"time"

	"github.com/samber/oops"
)

// Wire layout: [priority:1][timestamp:8][produceTime:8][appendTime:8][keyLen:4][key][value]
const (
	priorityFieldSize  = 1
	timestampFieldSize = 8
	keyLenFieldSize    = 4

	priorityOffset = 0                                   // 0
	timestampOff   = priorityFieldSize                   // 1
	produceTimeOff = timestampOff + timestampFieldSize   // 9
	appendTimeOff  = produceTimeOff + timestampFieldSize // 17
	keyLenOff      = appendTimeOff + timestampFieldSize  // 25
	keyOff         = keyLenOff + keyLenFieldSize         // 29

	// HeaderSize is the fixed-width prefix before key length (priority + 3 timestamps).
	HeaderSize = keyLenOff // 25

	// MetadataSize includes all fixed fields including key length.
	MetadataSize = keyOff // 29
)

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
	BatchDecoded bool `json:"-"`
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
	kl := len(m.Key)
	buf[priorityOffset] = m.Header.Priority
	binary.BigEndian.PutUint64(buf[timestampOff:], uint64(m.Header.Timestamp))
	binary.BigEndian.PutUint64(buf[produceTimeOff:], uint64(m.Header.ProduceTime))
	binary.BigEndian.PutUint64(buf[appendTimeOff:], uint64(m.Header.AppendTime))
	binary.BigEndian.PutUint32(buf[keyLenOff:], uint32(kl))
	copy(buf[keyOff:keyOff+kl], m.Key)
	copy(buf[keyOff+kl:], m.Value)
}

func UnmarshalMessage(data []byte) (*Message, error) {
	if len(data) < MetadataSize {
		return nil, oops.Errorf("message data too short: got %d bytes, need at least %d", len(data), MetadataSize)
	}

	kl := int(binary.BigEndian.Uint32(data[keyLenOff:]))
	if len(data) < MetadataSize+kl {
		return nil, oops.Errorf("message truncated: got %d bytes, need %d (keyLen=%d)", len(data), MetadataSize+kl, kl)
	}

	key := make([]byte, kl)
	copy(key, data[keyOff:keyOff+kl])

	value := make([]byte, len(data)-(keyOff+kl))
	copy(value, data[keyOff+kl:])

	return &Message{
		Header: MessageHeader{
			Priority:    data[priorityOffset],
			Timestamp:   int64(binary.BigEndian.Uint64(data[timestampOff:])),
			ProduceTime: int64(binary.BigEndian.Uint64(data[produceTimeOff:])),
			AppendTime:  int64(binary.BigEndian.Uint64(data[appendTimeOff:])),
		},
		Key:   key,
		Value: value,
	}, nil
}

func UnmarshalHeader(data []byte) (MessageHeader, error) {
	if len(data) < HeaderSize {
		return MessageHeader{}, oops.Errorf("data too short for header: got %d bytes, need at least %d", len(data), HeaderSize)
	}

	return MessageHeader{
		Priority:    data[priorityOffset],
		Timestamp:   int64(binary.BigEndian.Uint64(data[timestampOff:])),
		ProduceTime: int64(binary.BigEndian.Uint64(data[produceTimeOff:])),
		AppendTime:  int64(binary.BigEndian.Uint64(data[appendTimeOff:])),
	}, nil
}
