package algebrain

import (
	"io/ioutil"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	CharCount  = 128
	Terminator = 0

	encoderOutSize = 100
	focusInfoSize  = 50

	maxResponseLen = 1000
)

func init() {
	var n Network
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetwork)
}

// A Network uses a machine translation architecture to
// produce output expressions for input queries.
type Network struct {
	Encoder   seqfunc.Func
	Decoder   rnn.Block
	Attention neuralnet.Network
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var res Network
	err := serializer.DeserializeAny(d, &res.Encoder, &res.Decoder, &res.Attention)
	if err != nil {
		return nil, err
	}
	return &res, nil
}

// NewNetwork creates a randomly-initialized Network.
func NewNetwork() *Network {
	encoder := &rnn.Bidirectional{
		Forward: &rnn.BlockSeqFunc{
			B: rnn.StackedBlock{
				rnn.NewLSTM(CharCount, 200),
				rnn.NewLSTM(200, encoderOutSize),
			},
		},
		Backward: &rnn.BlockSeqFunc{
			B: rnn.StackedBlock{
				rnn.NewLSTM(CharCount, 200),
				rnn.NewLSTM(200, encoderOutSize),
			},
		},
		Output: &rnn.BlockSeqFunc{
			B: newOutputBlock(encoderOutSize*2, encoderOutSize, nil),
		},
	}
	decoderBlock := rnn.StackedBlock{
		rnn.NewLSTM(encoderOutSize, 200),
		rnn.NewLSTM(200, focusInfoSize+CharCount),
	}
	attention := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  focusInfoSize + encoderOutSize,
			OutputCount: 100,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 1,
		},
	}
	attention.Randomize()
	return &Network{
		Encoder:   encoder,
		Decoder:   decoderBlock,
		Attention: attention,
	}
}

// LoadNetwork loads a Network from a file.
func LoadNetwork(path string) (*Network, error) {
	contents, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DeserializeNetwork(contents)
}

// Parameters gets the parameters of the block.
func (n *Network) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, block := range []interface{}{n.Encoder, n.Decoder, n.Attention} {
		if l, ok := block.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
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
	return serializer.SerializeAny(n.Encoder, n.Decoder, n.Attention)
}

// Save writes the Network to a file.
func (n *Network) Save(path string) error {
	enc, err := n.Serialize()
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, enc, 0755)
}

// Query runs a query against this Network.
func (n *Network) Query(q string) string {
	sample := Sample{Query: q}
	encoded := n.Encoder.ApplySeqs(sample.InputSequence())
	decoder := n.evaluationBlock(encoded, nil)

	r := &rnn.Runner{Block: decoder}
	r.StepTime(zeroVector())

	var lastOut rune = Terminator
	var res string
	for {
		nextVec := r.StepTime(oneHotVector(lastOut))
		_, nextIdx := nextVec.Max()
		lastOut = rune(nextIdx)
		if lastOut == 0 || len(res) >= maxResponseLen {
			break
		}
		res += string(lastOut)
	}
	return res
}

func (n *Network) evaluationBlock(enc seqfunc.Result, g autofunc.Gradient) rnn.Block {
	s := attentor{
		Encoded: enc,
		Network: n.Attention,
		Grad:    g,
	}
	return rnn.StackedBlock{
		&neuralstruct.Block{
			Struct: &s,
			Block:  n.Decoder,
		},
		rnn.NewNetworkBlock(neuralnet.Network{&neuralnet.LogSoftmaxLayer{}}, 0),
	}
}

func newOutputBlock(inCount, outCount int, activation neuralnet.Layer) rnn.Block {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  inCount,
			OutputCount: outCount,
		},
	}
	if activation != nil {
		net = append(net, activation)
	}
	net.Randomize()
	return rnn.NewNetworkBlock(net, 0)
}
