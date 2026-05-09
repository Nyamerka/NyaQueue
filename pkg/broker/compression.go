package broker

import (
	"bytes"
	"compress/gzip"
	"io"

	"github.com/klauspost/compress/snappy"
	"github.com/pierrec/lz4/v4"
)

const (
	CompressionNone   = 0
	CompressionSnappy = 1
	CompressionGzip   = 2
	CompressionLZ4    = 3
)

func Compress(data []byte, codec int) ([]byte, error) {
	switch codec {
	case CompressionSnappy:
		return snappy.Encode(nil, data), nil
	case CompressionGzip:
		var buf bytes.Buffer
		w := gzip.NewWriter(&buf)
		if _, err := w.Write(data); err != nil {
			return nil, err
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	case CompressionLZ4:
		var buf bytes.Buffer
		w := lz4.NewWriter(&buf)
		if _, err := w.Write(data); err != nil {
			return nil, err
		}
		if err := w.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	default:
		return data, nil
	}
}

func Decompress(data []byte, codec int) ([]byte, error) {
	switch codec {
	case CompressionSnappy:
		return snappy.Decode(nil, data)
	case CompressionGzip:
		r, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		defer r.Close()
		return io.ReadAll(r)
	case CompressionLZ4:
		r := lz4.NewReader(bytes.NewReader(data))
		return io.ReadAll(r)
	default:
		return data, nil
	}
}
