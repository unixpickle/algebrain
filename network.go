package algebrain

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/attention"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

const (
	CharCount  = 0x80
	Terminator = 0

	querySize     = 0x80
	encodedSize   = 0x40
	decoderInSize = CharCount + encodedSize

	maxResponseLen = 0x400
)

func init() {
	var n Network
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetwork)
}

// A Network uses a machine translation architecture to
// produce output expressions for input queries.
type Network struct {
	Encoder *anyrnn.Bidir
	Align   *attention.SoftAlign
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var res Network
	if err := serializer.DeserializeAny(d, &res.Encoder, &res.Align); err != nil {
		return nil, essentials.AddCtx("deserialize Network", err)
	}
	return &res, nil
}

// NewNetwork creates a randomly-initialized Network.
func NewNetwork(c anyvec.Creator) *Network {
	inScaler := c.MakeNumeric(16)
	encoder := &anyrnn.Bidir{
		Forward: anyrnn.Stack{
			anyrnn.NewLSTM(c, CharCount, 0x100).ScaleInWeights(inScaler),
			anyrnn.NewLSTM(c, 0x100, encodedSize),
		},
		Backward: anyrnn.Stack{
			anyrnn.NewLSTM(c, CharCount, 0x100).ScaleInWeights(inScaler),
			anyrnn.NewLSTM(c, 0x100, encodedSize),
		},
		Mixer: &anynet.AddMixer{
			In1: anynet.NewFC(c, encodedSize, encodedSize),
			In2: anynet.NewFC(c, encodedSize, encodedSize),
			Out: anynet.Tanh,
		},
	}
	decoderBlock := anyrnn.Stack{
		anyrnn.NewLSTM(c, decoderInSize, 0x100),
		anyrnn.NewLSTM(c, 0x100, querySize),
	}
	inComb := &anynet.AddMixer{
		In1: anynet.NewFC(c, encodedSize, decoderInSize),
		In2: anynet.NewFC(c, CharCount, decoderInSize),
		Out: anynet.Tanh,
	}
	inComb.In2.(*anynet.FC).Weights.Vector.Scale(inScaler)
	attentor := &anynet.AddMixer{
		In1: anynet.NewFC(c, querySize, 0x80),
		In2: anynet.NewFC(c, encodedSize, 0x80),
		Out: anynet.Net{
			anynet.Tanh,
			anynet.NewFC(c, 0x80, 1),
			&anynet.Affine{
				Scalers: anydiff.NewVar(c.MakeVectorData(c.MakeNumericList([]float64{5}))),
				Biases:  anydiff.NewVar(c.MakeVectorData(c.MakeNumericList([]float64{0}))),
			},
		},
	}
	return &Network{
		Encoder: encoder,
		Align: &attention.SoftAlign{
			Attentor:   attentor,
			Decoder:    decoderBlock,
			InCombiner: inComb,
			InitQuery:  anydiff.NewVar(c.MakeVector(querySize)),
		},
	}
}

// Parameters gets the parameters of the network.
func (n *Network) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, p := range []anynet.Parameterizer{n.Encoder, n.Align} {
		res = append(res, p.Parameters()...)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Network with the serializer package.
func (n *Network) SerializerType() string {
	return "github.com/unixpickle/algebrain.Network"
}

// Serialize attempts to serialize the Network.
func (n *Network) Serialize() ([]byte, error) {
	return serializer.SerializeAny(n.Encoder, n.Align)
}

// Query runs a query against this Network.
func (n *Network) Query(q string) string {
	sample := Sample{Query: q}
	inSeq := anyseq.ConstSeqList([][]anyvec.Vector{sample.InputSequence()})
	enc := n.Encoder.Apply(inSeq)
	b := anyrnn.Stack{
		n.Align.Block(enc),
		&anyrnn.LayerBlock{Layer: anynet.LogSoftmax},
	}
	state := b.Start(1)

	var lastChar rune
	var res string

	for {
		result := b.Step(state, oneHotVector(lastChar))
		state = result.State()
		nextIdx := anyvec.MaxIndex(result.Output())
		lastChar = rune(nextIdx)
		if lastChar == 0 || len(res) >= maxResponseLen {
			break
		}
		res += string(lastChar)
	}

	return res
}
