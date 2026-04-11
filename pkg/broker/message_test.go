package broker

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type MessageSuite struct {
	suite.Suite
}

func TestMessageSuite(t *testing.T) { suite.Run(t, new(MessageSuite)) }

func (s *MessageSuite) TestMarshalUnmarshal() {
	tests := []struct {
		name     string
		priority uint8
		key      []byte
		value    []byte
	}{
		{"empty", 0, nil, nil},
		{"simple", 5, []byte("k"), []byte("v")},
		{"high_priority", 9, []byte("key-abc"), []byte("hello world")},
		{"large_value", 3, []byte("k"), make([]byte, 4096)},
		{"empty_key", 1, nil, []byte("val")},
		{"empty_value", 7, []byte("key"), nil},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			msg := NewMessage(tc.priority, tc.key, tc.value)
			require.Equal(s.T(), tc.priority, msg.Header.Priority)
			require.NotZero(s.T(), msg.Header.Timestamp)

			data := msg.Marshal()
			require.True(s.T(), len(data) >= HeaderSize+4)

			restored, err := UnmarshalMessage(data)
			require.NoError(s.T(), err)
			require.Equal(s.T(), tc.priority, restored.Header.Priority)
			require.Equal(s.T(), msg.Header.Timestamp, restored.Header.Timestamp)
			require.Equal(s.T(), len(tc.key), len(restored.Key))
			require.Equal(s.T(), len(tc.value), len(restored.Value))
			if len(tc.key) > 0 {
				require.Equal(s.T(), tc.key, restored.Key)
			}
			if len(tc.value) > 0 {
				require.Equal(s.T(), tc.value, restored.Value)
			}
		})
	}
}

func (s *MessageSuite) TestUnmarshalHeaderOnly() {
	msg := NewMessage(7, []byte("k"), []byte("v"))
	data := msg.Marshal()

	hdr, err := UnmarshalHeader(data)
	require.NoError(s.T(), err)
	require.Equal(s.T(), uint8(7), hdr.Priority)
	require.Equal(s.T(), msg.Header.Timestamp, hdr.Timestamp)
}

func (s *MessageSuite) TestUnmarshalErrors() {
	tests := []struct {
		name string
		data []byte
	}{
		{"nil", nil},
		{"too_short", []byte{1, 2, 3}},
		{"header_only", make([]byte, HeaderSize)},
	}

	for _, tc := range tests {
		s.Run(tc.name, func() {
			_, err := UnmarshalMessage(tc.data)
			require.Error(s.T(), err)
		})
	}
}

func (s *MessageSuite) TestNewMessageTimestamp() {
	before := time.Now().UnixNano()
	msg := NewMessage(0, nil, nil)
	after := time.Now().UnixNano()

	require.GreaterOrEqual(s.T(), msg.Header.Timestamp, before)
	require.LessOrEqual(s.T(), msg.Header.Timestamp, after)
}
