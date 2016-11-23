package algebrain

import (
	"fmt"
	"math/rand"

	"github.com/unixpickle/algebrain/mathexpr"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Sample contains a query (e.g. "factorize x^2+x") and
// the expected result (e.g. "x(x+1)").
type Sample struct {
	Query    string
	Response string
}

// InputSequence generates the sample's input sequence.
func (s *Sample) InputSequence() seqfunc.Result {
	res := make([]linalg.Vector, len(s.Query))
	for i, x := range s.Query {
		res[i] = oneHotVector(x)
	}
	return seqfunc.ConstResult([][]linalg.Vector{res})
}

// OutputSequence generates the sample's output sequence.
func (s *Sample) OutputSequence() seqfunc.Result {
	res := make([]linalg.Vector, len(s.Response)+1)
	res[0] = zeroVector()
	for i, x := range s.Response {
		res[i+1] = oneHotVector(x)
	}
	return seqfunc.ConstResult([][]linalg.Vector{res})
}

// A Generator generates random Samples from a template.
type Generator interface {
	Generate() *Sample
}

// A ShiftGenerator generates Samples with queries like
// "shift x by 2 in x^2+2", producing results like
// "(x-2)^2+2".
type ShiftGenerator struct {
	Generator *mathexpr.Generator
	MaxDepth  int
}

// Generate generates a graph shifting sample.
func (s *ShiftGenerator) Generate() *Sample {
	expr := s.Generator.Generate(s.MaxDepth)
	shiftVar := s.Generator.VarNames[rand.Intn(len(s.Generator.VarNames))]
	num := generateNumber(*s.Generator)
	query := fmt.Sprintf("shift %s by %s in %s", shiftVar, num, expr)
	output := s.shiftNode(shiftVar, num, expr).String()
	return &Sample{
		Query:    query,
		Response: output,
	}
}

func (s *ShiftGenerator) shiftNode(varName string, amount mathexpr.Node,
	n mathexpr.Node) mathexpr.Node {
	if n, ok := n.(mathexpr.RawNode); ok {
		if string(n) == varName {
			return &mathexpr.BinaryOp{
				Op:    mathexpr.SubtractOp,
				Left:  n,
				Right: amount,
			}
		}
	}
	for i, x := range n.Children() {
		n.SetChild(i, s.shiftNode(varName, amount, x))
	}
	return n
}

// A ScaleGenerator generates Samples with queries like
// "scale x by 2 in x^2", expecting "(2*x)^2".
type ScaleGenerator struct {
	Generator *mathexpr.Generator
	MaxDepth  int
}

func (s *ScaleGenerator) Generate() *Sample {
	expr := s.Generator.Generate(s.MaxDepth)
	shiftVar := s.Generator.VarNames[rand.Intn(len(s.Generator.VarNames))]
	num := generateNumber(*s.Generator)
	query := fmt.Sprintf("scale %s by %s in %s", shiftVar, num, expr)
	output := s.scaleNode(shiftVar, num, expr).String()
	return &Sample{
		Query:    query,
		Response: output,
	}
}

func (s *ScaleGenerator) scaleNode(varName string, amount mathexpr.Node,
	n mathexpr.Node) mathexpr.Node {
	if n, ok := n.(mathexpr.RawNode); ok {
		if string(n) == varName {
			return &mathexpr.BinaryOp{
				Op:    mathexpr.MultiplyOp,
				Left:  n,
				Right: amount,
			}
		}
	}
	for i, x := range n.Children() {
		n.SetChild(i, s.scaleNode(varName, amount, x))
	}
	return n
}

func generateNumber(g mathexpr.Generator) mathexpr.RawNode {
	g.VarNames = nil
	g.ConstNames = nil
	return g.Generate(0).(mathexpr.RawNode)
}

func zeroVector() linalg.Vector {
	return make(linalg.Vector, CharCount)
}

func oneHotVector(x rune) linalg.Vector {
	res := make(linalg.Vector, CharCount)
	ix := int(x)
	if ix > 128 || ix < 0 {
		panic("rune out of range: " + string(x))
	}
	res[ix] = 1
	return res
}
