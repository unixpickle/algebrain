package mathexpr

type Precedence int

const (
	AddPrecedence Precedence = iota
	MultPrecedence
	NegPrecedence
	ExpPrecedence
	AtomicPrecedence
)

// A Node is an algebraic sub-expression in a tree.
type Node interface {
	// Precedence returns the maximum free-floating
	// Precedence in the expression.
	// In other words, it describes the strength of the glue
	// that holds the expression together.
	//
	// For instance, an expression "3+2*3" would give
	// AddPrecedence, while "(3+2)*3" would give
	// MultPrecedence.
	Precedence() Precedence

	// String returns the expression's string representation.
	String() string

	// Children returns the node's children.
	// This slice should be a copy, meaning that the caller
	// may modify it.
	Children() []Node

	// SetChild updates the n-th child.
	SetChild(n int, c Node)
}
