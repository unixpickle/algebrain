package algebrain

import (
	"encoding/json"
	"io/ioutil"

	"github.com/unixpickle/attention"
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	CharCount  = 128
	Terminator = 0

	encoderOutSize     = 15
	encoderStateSize   = 30
	focusInfoSize      = 30
	attentionBatchSize = 32

	maxResponseLen = 1000
)

func init() {
	var n Network
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNetwork)
}

// A Network uses a machine translation architecture to
// produce output expressions for input queries.
type Network struct {
	Encoder   seqfunc.RFunc
	Decoder   rnn.Block
	Attention neuralnet.Network
	InitQuery *autofunc.Variable
}

// DeserializeNetwork deserializes a Network.
func DeserializeNetwork(d []byte) (*Network, error) {
	var res Network
	var initData serializer.Bytes
	err := serializer.DeserializeAny(d, &res.Encoder, &res.Decoder, &res.Attention, &initData)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(initData, &res.InitQuery); err != nil {
		return nil, err
	}
	return &res, nil
}

// NewNetwork creates a randomly-initialized Network.
func NewNetwork() *Network {
	encoder := &rnn.Bidirectional{
		Forward: &rnn.BlockSeqFunc{B: inCharScale(&rnn.StateOutBlock{
			Block: newNetworkBlock(CharCount+encoderStateSize, encoderStateSize,
				encoderStateSize, neuralnet.HyperbolicTangent{}),
		})},
		Backward: &rnn.BlockSeqFunc{B: inCharScale(&rnn.StateOutBlock{
			Block: newNetworkBlock(CharCount+encoderStateSize, encoderStateSize,
				encoderStateSize, neuralnet.HyperbolicTangent{}),
		})},
		Output: &rnn.BlockSeqFunc{
			B: newOutputBlock(encoderStateSize*2, encoderOutSize,
				neuralnet.HyperbolicTangent{}),
		},
	}
	decoderBlock := rnn.StackedBlock{
		rnn.NewLSTM(encoderOutSize+CharCount, 300),
		rnn.NewLSTM(300, 300),
		newOutputBlock(300, focusInfoSize+CharCount, &neuralstruct.PartialActivation{
			Ranges: []neuralstruct.ComponentRange{
				{Start: 0, End: focusInfoSize},
			},
			Activations: []neuralnet.Layer{&neuralnet.HyperbolicTangent{}},
		}),
	}
	attention := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  focusInfoSize + encoderOutSize,
			OutputCount: 300,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  300,
			OutputCount: 100,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 1,
		},
		&neuralnet.RescaleLayer{Scale: 1},
	}
	attention.Randomize()
	return &Network{
		Encoder:   encoder,
		Decoder:   decoderBlock,
		Attention: attention,
		InitQuery: &autofunc.Variable{Vector: make(linalg.Vector, focusInfoSize)},
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
	res := []*autofunc.Variable{n.InitQuery}
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
	data, err := json.Marshal(n.InitQuery)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(n.Encoder, n.Decoder, n.Attention, serializer.Bytes(data))
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
	r := n.softAlign().TimeStepper(sample.InputSequence())

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
	cost := n.computeCost(s)
	cost.PropagateGradient([]float64{1}, grad)
	return grad
}

// TotalCost computes the total cost for a batch.
func (n *Network) TotalCost(s sgd.SampleSet) float64 {
	return n.computeCost(s).Output()[0]
}

func (n *Network) computeCost(s sgd.SampleSet) autofunc.Result {
	var inSeqs [][]linalg.Vector
	var outSeqs [][]linalg.Vector
	var decInSeqs [][]linalg.Vector
	for i := 0; i < s.Len(); i++ {
		s := s.GetSample(i).(*Sample)
		inSeqs = append(inSeqs, s.InputSequence())
		outSeqs = append(outSeqs, s.DecoderOutSequence())
		decInSeqs = append(decInSeqs, s.DecoderInSequence())
	}
	output := n.softAlign().Apply(seqfunc.ConstResult(inSeqs),
		seqfunc.ConstResult(decInSeqs))
	return seqfunc.AddAll(seqfunc.MapN(func(ins ...autofunc.Result) autofunc.Result {
		actual := ins[0]
		expected := ins[1].Output()
		cf := &neuralnet.DotCost{}
		logSoft := neuralnet.LogSoftmaxLayer{}
		return cf.Cost(expected, logSoft.Apply(actual))
	}, output, seqfunc.ConstResult(outSeqs)))
}

func (n *Network) softAlign() *attention.SoftAlign {
	return &attention.SoftAlign{
		Encoder:    n.Encoder,
		Attentor:   n.Attention,
		Decoder:    n.Decoder,
		BatchSize:  attentionBatchSize,
		StartQuery: n.InitQuery,
	}
}

func newOutputBlock(inCount, outCount int, activation neuralnet.Layer) rnn.Block {
	return newNetworkBlock(inCount, outCount, 0, activation)
}

func newNetworkBlock(inCount, outCount, state int, activation neuralnet.Layer) rnn.Block {
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
	biases := net[0].(*neuralnet.DenseLayer).Biases
	biases.Var.Vector.Scale(0)
	return rnn.NewNetworkBlock(net, state)
}

func inCharScale(b rnn.Block) rnn.Block {
	return rnn.StackedBlock{
		rnn.NewNetworkBlock(neuralnet.Network{
			&neuralnet.RescaleLayer{Scale: 10},
		}, 0),
		b,
	}
}
