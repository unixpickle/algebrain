package algebrain

import (
	"fmt"
	"math/rand"

	"github.com/unixpickle/algebrain/mathexpr"
)

// A Sample contains a query (e.g. "factorize x^2+x") and
// the expected result (e.g. "x(x+1)").
type Sample struct {
	Query    string
	Response string
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
