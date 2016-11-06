package algebrain

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/rnntest"
)

const testCharCount = 4

func TestBlock(t *testing.T) {
	if testing.Short() {
		t.SkipNow()
	}

	inSeqs := [][]*autofunc.Variable{
		testingString([]int{1, 2, 0, 1, 3, 2}),
		testingString([]int{2, 0, 1, 2}),
		testingString([]int{2, 0, 1, 2, 3, 2, 1}),
	}
	block := &Block{
		Reader: rnn.NewLSTM(testCharCount, testCharCount),
		Writer: rnn.NewLSTM(testCharCount, testCharCount),
	}

	params := block.Parameters()
	for _, seq := range inSeqs {
		for _, vec := range seq {
			if vec.Vector[Terminator] != 0 {
				params = append(params, vec)
			}
		}
	}

	rv := autofunc.RVector{}
	for _, p := range params {
		rv[p] = make(linalg.Vector, len(p.Vector))
		for i := range rv[p] {
			rv[p][i] = rand.NormFloat64()
		}
	}

	checker := rnntest.BlockChecker{
		B:     block,
		Input: inSeqs,
		Vars:  params,
		RV:    rv,
	}
	checker.FullCheck(t)
}

func testingString(s []int) []*autofunc.Variable {
	res := make([]*autofunc.Variable, len(s))
	for i, x := range s {
		res[i] = testingOneHot(x)
	}
	return res
}

func testingOneHot(c int) *autofunc.Variable {
	res := &autofunc.Variable{Vector: make(linalg.Vector, testCharCount)}
	res.Vector[c] = 1
	return res
}
