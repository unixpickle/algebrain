package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/unixpickle/algebrain"
	"github.com/unixpickle/algebrain/mathexpr"
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

var Generators = map[string]algebrain.Generator{
	"EasyShift": &algebrain.ShiftGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x"},
		},
		MaxDepth: 1,
	},
	"EasyScale": &algebrain.ScaleGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x"},
		},
		MaxDepth: 1,
	},
	"MediumShift": &algebrain.ShiftGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x"},
		},
		MaxDepth: 3,
	},
	"MediumScale": &algebrain.ScaleGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x"},
		},
		MaxDepth: 3,
	},
}

var (
	NewBlockStruct = neuralstruct.RAggregate{
		&neuralstruct.Stack{VectorSize: 30, NoReplace: true, PushBias: 2},
		&neuralstruct.Stack{VectorSize: 30, NoReplace: true, PushBias: 2},
		&neuralstruct.Queue{VectorSize: 30, PushBias: 2},
		&neuralstruct.Queue{VectorSize: 30, PushBias: 2},
	}
	NewBlockHidden  = []int{512, 512}
	NewBlockDropout = 0.9
)

func main() {
	var genNames string
	var stepSize float64
	var batchSize int
	var outFile string
	var samplesPerGen int
	flag.StringVar(&genNames, "generators", "EasyShift,MediumShift,EasyScale,MediumScale",
		"comma-separated generator list")
	flag.Float64Var(&stepSize, "step", 0.005, "SGD step size")
	flag.IntVar(&batchSize, "batch", 4, "SGD batch size")
	flag.StringVar(&outFile, "file", "out_net", "output/input network file")
	flag.IntVar(&samplesPerGen, "samples", 10000, "samples per generator")
	flag.Parse()

	log.Println("Creating samples...")
	training, validation := generateSamples(genNames, samplesPerGen)

	rand.Seed(time.Now().UnixNano())

	log.Println("Obtaining RNN block...")
	block, err := algebrain.LoadBlock(outFile)
	if os.IsNotExist(err) {
		log.Println("Creating new RNN block...")
		block = algebrain.NewBlock(NewBlockDropout, NewBlockStruct, NewBlockHidden...)
	} else if err != nil {
		die("Failed to load block:", err)
	}

	log.Println("Training...")

	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &seqtoseq.Gradienter{
			SeqFunc:  &rnn.BlockSeqFunc{B: block},
			Learner:  block,
			CostFunc: costFunc,
			MaxLanes: batchSize,
		},
	}

	var lastBatch sgd.SampleSet
	var iter int
	block.Dropout(true)
	sgd.SGDMini(gradienter, training, stepSize, batchSize, func(b sgd.SampleSet) bool {
		block.Dropout(false)
		defer block.Dropout(true)
		var lastCost float64
		if lastBatch != nil {
			lastCost = seqtoseq.TotalCostBlock(block, batchSize, lastBatch, costFunc)
		}
		lastBatch = b
		cost := seqtoseq.TotalCostBlock(block, batchSize, b, costFunc)

		sgd.ShuffleSampleSet(validation)
		subValidation := validation.Subset(0, batchSize)
		val := seqtoseq.TotalCostBlock(block, batchSize, subValidation, costFunc)
		log.Printf("iter %d: validation=%f cost=%f last=%f", iter, val, cost, lastCost)
		iter++
		return true
	})

	block.Dropout(false)
	if err := block.Save(outFile); err != nil {
		die("Failed to save block:", err)
	}
}

func generateSamples(genNames string, samplesPer int) (training, validation algebrain.SampleSet) {
	// Ensure that we get the same samples every time.
	rand.Seed(123)

	names := strings.Split(genNames, ",")
	gens := make([]algebrain.Generator, len(names))
	for i, x := range names {
		if gen, ok := Generators[x]; ok {
			gens[i] = gen
		} else {
			die("Unknown generator:", x)
		}
	}
	for _, g := range gens {
		for i := 0; i < samplesPer; i++ {
			training = append(training, g.Generate())
			validation = append(validation, g.Generate())
		}
	}
	return
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
