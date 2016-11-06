package main

import (
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/rnn"
)

func init() {
	var b Block
	serializer.RegisterDeserializer(b.SerializerType(), DeserializeBlock)
}

// A Block uses two RNNs to evaluate an algebra query.
type Block struct {
	Reader rnn.Block
	Writer rnn.Block
}

// DeserializeBlock deserializes a block.
func DeserializeBlock(d []byte) (*Block, error) {
	var res Block
	if err := serializer.DeserializeAny(d, &res.Reader, &res.Writer); err != nil {
		return nil, err
	}
	return &res, nil
}

// SerializerType returns the unique ID used to serialize
// a Block with the serializer package.
func (b *Block) SerializerType() string {
	return "github.com/unixpickle/algebrain.Block"
}

// Serialize attempts to serialize the block.
func (b *Block) Serialize() ([]byte, error) {
	return serializer.SerializeAny(b.Reader, b.Writer)
}
