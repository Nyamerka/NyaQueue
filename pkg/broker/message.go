package broker

import (
	"encoding/binary"
	"errors"
	"time"
)

const HeaderSize = 9 // 1 (priority) + 8 (timestamp)

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

// Marshal encodes message as:
//
//	[priority:1][timestamp:8][keyLen:4][key:...][value:...]
func (m *Message) Marshal() []byte {
	keyLen := len(m.Key)
	buf := make([]byte, HeaderSize+4+keyLen+len(m.Value))

	buf[0] = m.Header.Priority
	binary.BigEndian.PutUint64(buf[1:9], uint64(m.Header.Timestamp))
	binary.BigEndian.PutUint32(buf[9:13], uint32(keyLen))
	copy(buf[13:13+keyLen], m.Key)
	copy(buf[13+keyLen:], m.Value)

	return buf
}

func UnmarshalMessage(data []byte) (*Message, error) {
	if len(data) < HeaderSize+4 {
		return nil, errors.New("message data too short")
	}

	priority := data[0]
	timestamp := int64(binary.BigEndian.Uint64(data[1:9]))
	keyLen := int(binary.BigEndian.Uint32(data[9:13]))

	if len(data) < HeaderSize+4+keyLen {
		return nil, errors.New("message data truncated")
	}

	key := make([]byte, keyLen)
	copy(key, data[13:13+keyLen])

	value := make([]byte, len(data)-(13+keyLen))
	copy(value, data[13+keyLen:])

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
		Timestamp: int64(binary.BigEndian.Uint64(data[1:9])),
	}, nil
}
