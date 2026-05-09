package broker

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompressDecompressRoundTrip(t *testing.T) {
	payload := []byte("The quick brown fox jumps over the lazy dog. Repeat for compression ratio.")

	for _, codec := range []int{CompressionNone, CompressionSnappy, CompressionGzip, CompressionLZ4} {
		t.Run(compressionName(codec), func(t *testing.T) {
			compressed, err := Compress(payload, codec)
			require.NoError(t, err)

			decompressed, err := Decompress(compressed, codec)
			require.NoError(t, err)

			require.Equal(t, payload, decompressed)
		})
	}
}

func TestCompressNoneIsIdentity(t *testing.T) {
	data := []byte("unchanged")
	out, err := Compress(data, CompressionNone)
	require.NoError(t, err)
	require.Equal(t, data, out)

	back, err := Decompress(out, CompressionNone)
	require.NoError(t, err)
	require.Equal(t, data, back)
}

func TestCompressSnappyShrinks(t *testing.T) {
	payload := make([]byte, 4096)
	for i := range payload {
		payload[i] = 'A'
	}

	compressed, err := Compress(payload, CompressionSnappy)
	require.NoError(t, err)
	require.Less(t, len(compressed), len(payload), "snappy should compress repetitive data")
}
