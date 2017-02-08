package main

import (
	"flag"
	"log"
	"math/rand"
	"strings"
	"time"

	"github.com/unixpickle/algebrain"
	"github.com/unixpickle/algebrain/mathexpr"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
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
	"EasyEval": &algebrain.EvalGenerator{
		Generator: &mathexpr.Generator{
			NoReals: true,
		},
		MaxDepth: 1,
		AllInts:  true,
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
	"MediumEval": &algebrain.EvalGenerator{
		Generator: &mathexpr.Generator{
			NoReals: true,
			Stddev:  80,
		},
		MaxDepth: 3,
		AllInts:  true,
	},
	"HardShift": &algebrain.ShiftGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x", "y", "z"},
		},
		MaxDepth: 5,
	},
	"HardScale": &algebrain.ScaleGenerator{
		Generator: &mathexpr.Generator{
			NoReals:  true,
			VarNames: []string{"x", "y", "z"},
		},
		MaxDepth: 5,
	},
}

func main() {
	var genNames string
	var stepSize float64
	var batchSize int
	var outFile string
	var samplesPerGen int
	flag.StringVar(&genNames, "generators",
		"EasyShift,MediumShift,EasyScale,MediumScale,EasyEval,MediumEval,HardShift,HardScale",
		"comma-separated generator list")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&batchSize, "batch", 8, "SGD batch size")
	flag.StringVar(&outFile, "file", "out_net", "output/input network file")
	flag.IntVar(&samplesPerGen, "samples", 10000, "samples per generator")
	flag.Parse()

	log.Println("Creating samples...")
	training := generateSamples(genNames, samplesPerGen)

	rand.Seed(time.Now().UnixNano())

	var net *algebrain.Network
	if err := serializer.LoadAny(outFile, &net); err != nil {
		log.Println("Creating new RNN block...")
		net = algebrain.NewNetwork(anyvec32.CurrentCreator())
	} else {
		log.Println("Loaded existing RNN block.")
	}

	log.Println("Training...")
	trainer := &algebrain.Trainer{Network: net}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     trainer,
		Gradienter:  trainer,
		Transformer: &anysgd.Adam{},
		Samples:     training,
		Rater:       anysgd.ConstRater(stepSize),
		BatchSize:   batchSize,
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iter, trainer.LastCost)
			iter++
		},
	}
	sgd.Run(rip.NewRIP().Chan())

	if err := serializer.SaveAny(outFile, net); err != nil {
		essentials.Die("Failed to save block:", err)
	}
}

func generateSamples(genNames string, samplesPer int) algebrain.SampleList {
	// Ensure that we get the same samples every time.
	rand.Seed(123)

	names := strings.Split(genNames, ",")
	gens := make([]algebrain.Generator, len(names))
	for i, x := range names {
		if gen, ok := Generators[x]; ok {
			gens[i] = gen
		} else {
			essentials.Die("Unknown generator:", x)
		}
	}
	var training algebrain.SampleList
	for _, g := range gens {
		for i := 0; i < samplesPer; i++ {
			training = append(training, g.Generate())
		}
	}
	return training
}
