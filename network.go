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
		rnn.NewLSTM(encoderOutSize+CharCount, 300),
		newOutputBlock(300, focusInfoSize+CharCount, nil),
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

// Gradient computes the cost gradient for a batch.
func (n *Network) Gradient(s sgd.SampleSet) autofunc.Gradient {
	grad := autofunc.NewGradient(n.Parameters())
	for i := 0; i < s.Len(); i++ {
		cost := n.computeCost(s.GetSample(i).(*Sample), grad)
		cost.PropagateGradient([]float64{1}, grad)
	}
	return grad
}

// TotalCost computes the total cost for a batch.
func (n *Network) TotalCost(s sgd.SampleSet) float64 {
	var cost float64
	for i := 0; i < s.Len(); i++ {
		c := n.computeCost(s.GetSample(i).(*Sample), nil)
		cost += c.Output()[0]
	}
	return cost
}

func (n *Network) computeCost(s *Sample, g autofunc.Gradient) autofunc.Result {
	encoded := n.Encoder.ApplySeqs(s.InputSequence())
	costs := seqfunc.Pool(encoded, func(encoded seqfunc.Result) seqfunc.Result {
		decoder := n.evaluationBlock(encoded, g)
		decoderSF := rnn.BlockSeqFunc{B: decoder}
		output := decoderSF.ApplySeqs(s.DecoderInSequence())
		return seqfunc.MapN(func(ins ...autofunc.Result) autofunc.Result {
			actual := ins[0]
			expected := ins[1].Output()
			cf := &neuralnet.DotCost{}
			return cf.Cost(expected, actual)
		}, output, s.DecoderOutSequence())
	})
	return seqfunc.AddAll(costs)
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
