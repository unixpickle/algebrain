package algebrain

import (
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// NewBlock creates a block that uses a multi-layer LSTM
// and an assortment of differentiable memory structures.
func NewBlock(dropout float64, structure neuralstruct.RAggregate, hiddenSizes ...int) *Block {
	resBlocks := make([]rnn.Block, 2)
	for i := range resBlocks {
		var sb rnn.StackedBlock
		inCount := structure.DataSize() + CharCount
		for _, hidden := range hiddenSizes {
			sb = append(sb, rnn.NewLSTM(inCount, hidden))
			sb = append(sb, rnn.NewNetworkBlock(neuralnet.Network{
				&neuralnet.DropoutLayer{
					KeepProbability: dropout,
				},
			}, 0))
			inCount = hidden
		}

		outNet := neuralnet.Network{
			&neuralnet.DenseLayer{
				InputCount:  inCount,
				OutputCount: CharCount + structure.ControlSize(),
			},
			structDataActivation(structure),
		}
		outNet.Randomize()
		sb = append(sb, rnn.NewNetworkBlock(outNet, 0))

		resBlocks[i] = &neuralstruct.Block{
			Struct: structure,
			Block:  sb,
		}
	}
	return &Block{Reader: resBlocks[0], Writer: resBlocks[1]}
}

func structDataActivation(ag neuralstruct.RAggregate) neuralnet.Layer {
	var activation neuralstruct.PartialActivation
	var idx int
	for _, subStruct := range ag {
		r := neuralstruct.ComponentRange{
			Start: idx + subStruct.ControlSize() - subStruct.DataSize(),
			End:   idx + subStruct.ControlSize(),
		}
		activation.Ranges = append(activation.Ranges, r)
		activation.Activations = append(activation.Activations,
			&neuralnet.HyperbolicTangent{})
		idx = r.End
	}
	activation.Ranges = append(activation.Ranges, neuralstruct.ComponentRange{
		Start: idx,
		End:   idx + CharCount,
	})
	activation.Activations = append(activation.Activations, &neuralnet.LogSoftmaxLayer{})
	return &activation
}
