package broker

import (
	"bytes"
	"compress/gzip"
	"io"
	"sync"

	"github.com/klauspost/compress/snappy"
	"github.com/pierrec/lz4/v4"
)

const (
	CompressionNone   = 0
	CompressionSnappy = 1
	CompressionGzip   = 2
	CompressionLZ4    = 3
)

type Codec interface {
	Encode([]byte) ([]byte, error)
	Decode([]byte) ([]byte, error)
}

func NewCodec(compressionType int) Codec {
	switch compressionType {
	case CompressionSnappy:
		return snappyCodec{}
	case CompressionGzip:
		return &gzipCodec{}
	case CompressionLZ4:
		return lz4Codec{}
	default:
		return noopCodec{}
	}
}

type noopCodec struct{}

func (noopCodec) Encode(data []byte) ([]byte, error) { return data, nil }
func (noopCodec) Decode(data []byte) ([]byte, error) { return data, nil }

type snappyCodec struct{}

func (snappyCodec) Encode(data []byte) ([]byte, error) {
	return snappy.Encode(nil, data), nil
}

func (snappyCodec) Decode(data []byte) ([]byte, error) {
	return snappy.Decode(nil, data)
}

var gzipWriterPool = sync.Pool{
	New: func() any { return gzip.NewWriter(io.Discard) },
}

var bufferPool = sync.Pool{
	New: func() any { return new(bytes.Buffer) },
}

type gzipCodec struct{}

func (g *gzipCodec) Encode(data []byte) ([]byte, error) {
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	w := gzipWriterPool.Get().(*gzip.Writer)
	w.Reset(buf)

	if _, err := w.Write(data); err != nil {
		w.Reset(io.Discard)
		gzipWriterPool.Put(w)
		bufferPool.Put(buf)
		return nil, err
	}
	if err := w.Close(); err != nil {
		w.Reset(io.Discard)
		gzipWriterPool.Put(w)
		bufferPool.Put(buf)
		return nil, err
	}

	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())

	w.Reset(io.Discard)
	gzipWriterPool.Put(w)
	bufferPool.Put(buf)
	return result, nil
}

func (g *gzipCodec) Decode(data []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return io.ReadAll(r)
}

type lz4Codec struct{}

func (lz4Codec) Encode(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	w := lz4.NewWriter(&buf)
	if _, err := w.Write(data); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (lz4Codec) Decode(data []byte) ([]byte, error) {
	r := lz4.NewReader(bytes.NewReader(data))
	return io.ReadAll(r)
}

func Compress(data []byte, codec int) ([]byte, error) {
	return NewCodec(codec).Encode(data)
}

func Decompress(data []byte, codec int) ([]byte, error) {
	return NewCodec(codec).Decode(data)
}
