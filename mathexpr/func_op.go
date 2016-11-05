package mathexpr

import "strings"

// FuncOp represents a call to a mathematical function.
type FuncOp struct {
	Name string
	Args []Node
}

// Precedence returns AtomicPrecedence.
func (f *FuncOp) Precedence() Precedence {
	return AtomicPrecedence
}

// String returns a string for the function call.
func (f *FuncOp) String() string {
	argsStr := make([]string, len(f.Args))
	for i, x := range f.Args {
		argsStr[i] = x.String()
	}
	return f.Name + "(" + strings.Join(argsStr, ", ") + ")"
}

// Children returns the list of child nodes.
func (f *FuncOp) Children() []Node {
	return append([]Node{}, f.Args...)
}

// SetChild sets an argument.
func (f *FuncOp) SetChild(n int, c Node) {
	f.Args[n] = c
}
