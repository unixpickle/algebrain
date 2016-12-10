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
	"github.com/unixpickle/sgd"
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
	var logInterval int
	flag.StringVar(&genNames, "generators",
		"EasyShift,MediumShift,EasyScale,MediumScale,EasyEval,MediumEval,HardShift,HardScale",
		"comma-separated generator list")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")
	flag.IntVar(&batchSize, "batch", 4, "SGD batch size")
	flag.StringVar(&outFile, "file", "out_net", "output/input network file")
	flag.IntVar(&samplesPerGen, "samples", 10000, "samples per generator")
	flag.IntVar(&logInterval, "logint", 4, "log interval")
	flag.Parse()

	log.Println("Creating samples...")
	training, validation := generateSamples(genNames, samplesPerGen)

	rand.Seed(time.Now().UnixNano())

	log.Println("Obtaining RNN block...")
	net, err := algebrain.LoadNetwork(outFile)
	if os.IsNotExist(err) {
		log.Println("Creating new RNN block...")
		net = algebrain.NewNetwork()
	} else if err != nil {
		die("Failed to load block:", err)
	}

	log.Println("Training...")

	gradienter := &sgd.RMSProp{Gradienter: net, Resiliency: 0.9}

	var lastBatch sgd.SampleSet
	var iter int
	sgd.SGDMini(gradienter, training, stepSize, batchSize, func(b sgd.SampleSet) bool {
		if iter%logInterval == 0 {
			var lastCost float64
			if lastBatch != nil {
				lastCost = net.TotalCost(lastBatch)
			}
			lastBatch = b
			cost := net.TotalCost(b)
			sgd.ShuffleSampleSet(validation)
			val := net.TotalCost(validation.Subset(0, batchSize))
			log.Printf("iter %d: validation=%f cost=%f last=%f", iter, val, cost, lastCost)
		}
		iter++
		return true
	})

	if err := net.Save(outFile); err != nil {
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
