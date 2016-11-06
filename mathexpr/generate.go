package mathexpr

import (
	"math/rand"
	"strconv"
)

// StandardFuncNames includes some standard function names
// in mathematics.
var StandardFuncNames = []string{"sin", "cos", "tan", "exp", "ln"}

// StandardConstNames includes some standard constants in
// mathematics.
var StandardConstNames = []string{"e", "pi"}

// DefaultGeneratorStddev is the default standard
// deviation used when generating random numbers.
const DefaultGeneratorStddev = 10

// A Generator generates random expressions.
type Generator struct {
	// If NoReals is set, all numerical constants will be
	// integers.
	NoReals bool

	// Stddev specifies the standard deviation for generating
	// random numbers on a normal distribution.
	// If this is 0, DefaultGeneratorStddev is used.
	Stddev float64

	// FuncNames stores the allowed function names.
	FuncNames []string

	// ConstNames stores the allowed constant names.
	ConstNames []string

	// VarNames stores the allowed variable names.
	VarNames []string
}

// Generate generates a random node with a given maximum
// nesting depth.
// If maxDepth is 0, the result must have no children.
func (g *Generator) Generate(maxDepth int) Node {
	if maxDepth == 0 || rand.Intn(maxDepth+1) == 0 {
		return g.randomRawNode()
	}
	if len(g.FuncNames) == 0 || rand.Intn(2) == 0 {
		return g.randomBinaryOp(maxDepth)
	}
	return g.randomFuncOp(maxDepth)
}

func (g *Generator) randomFuncOp(maxDepth int) *FuncOp {
	f := g.FuncNames[rand.Intn(len(g.FuncNames))]
	return &FuncOp{
		Name: f,
		Args: []Node{g.Generate(maxDepth - 1)},
	}
}

func (g *Generator) randomBinaryOp(maxDepth int) *BinaryOp {
	ops := []string{MultiplyOp, DivideOp, SubtractOp, AddOp, PowOp}
	op := ops[rand.Intn(len(ops))]
	return &BinaryOp{
		Op:    op,
		Left:  g.Generate(maxDepth - 1),
		Right: g.Generate(maxDepth - 1),
	}
}

func (g *Generator) randomRawNode() RawNode {
	options := []RawNode{}
	if len(g.ConstNames) > 0 {
		idx := rand.Intn(len(g.ConstNames))
		options = append(options, RawNode(g.ConstNames[idx]))
	}
	if len(g.VarNames) > 0 {
		idx := rand.Intn(len(g.VarNames))
		options = append(options, RawNode(g.VarNames[idx]))
	}
	options = append(options, g.randomNumNode())
	return options[rand.Intn(len(options))]
}

func (g *Generator) randomNumNode() RawNode {
	s := g.Stddev
	if s == 0 {
		s = DefaultGeneratorStddev
	}
	num := rand.NormFloat64() * s
	if g.NoReals {
		var intVal int
		if num < 0 {
			intVal = -int(-num + 0.5)
		} else {
			intVal = int(num + 0.5)
		}
		return RawNode(strconv.Itoa(intVal))
	} else {
		return RawNode(strconv.FormatFloat(num, 'f', -1, 64))
	}
}
