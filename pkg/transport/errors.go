package transport

import (
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/Nyamerka/NyaQueue/pkg/broker"
)

func mapBrokerError(err error) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, broker.ErrTopicAlreadyExists):
		return status.Error(codes.AlreadyExists, err.Error())
	case errors.Is(err, broker.ErrTopicNotFound):
		return status.Error(codes.NotFound, err.Error())
	}
	return err
}

func mapClientError(err error) error {
	if err == nil {
		return nil
	}
	switch status.Code(err) {
	case codes.AlreadyExists:
		return errors.Join(broker.ErrTopicAlreadyExists, err)
	case codes.NotFound:
		return errors.Join(broker.ErrTopicNotFound, err)
	}
	return err
}
