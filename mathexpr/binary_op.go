package mathexpr

import "strconv"

const (
	MultiplyOp = "*"
	DivideOp   = "/"
	SubtractOp = "-"
	AddOp      = "+"
	PowOp      = "^"
)

// BinaryOp is a binary operation between two nodes.
type BinaryOp struct {
	Left  Node
	Right Node
	Op    string
}

// Precedence returns the precedence corresponding to the
// binary operator.
func (b *BinaryOp) Precedence() Precedence {
	switch b.Op {
	case MultiplyOp, DivideOp:
		return MultPrecedence
	case AddOp, SubtractOp:
		return AddPrecedence
	case PowOp:
		return ExpPrecedence
	}
	panic("unknown operator: " + b.Op)
}

// String returns a string for the expression.
func (b *BinaryOp) String() string {
	left := b.Left.String()
	right := b.Right.String()
	if b.Left.Precedence() <= b.Precedence() {
		left = "(" + left + ")"
	}
	if b.Right.Precedence() <= b.Precedence() {
		right = "(" + right + ")"
	}
	return left + b.Op + right
}

// Children returns a slice with the left/right child.
func (b *BinaryOp) Children() []Node {
	return []Node{b.Left, b.Right}
}

// SetChild sets either the first or second child.
func (b *BinaryOp) SetChild(n int, c Node) {
	if n == 0 {
		b.Left = c
	} else if n == 1 {
		b.Right = c
	} else {
		panic("child out of bounds: " + strconv.Itoa(n))
	}
}
