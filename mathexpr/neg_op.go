package mathexpr

// A NegOp negates a Node.
type NegOp struct {
	Node Node
}

// Precedence returns NegPrecedence.
func (n *NegOp) Precedence() Precedence {
	return NegPrecedence
}

// String returns the expression's string representation.
func (n *NegOp) String() string {
	if n.Node.Precedence() <= NegPrecedence {
		return "-(" + n.Node.String() + ")"
	} else {
		return "-" + n.Node.String()
	}
}

// Children returns the node's only child.
func (n *NegOp) Children() []Node {
	return []Node{n.Node}
}

// SetChild updates the n-th child.
func (n *NegOp) SetChild(i int, c Node) {
	if i != 0 {
		panic("invalid index (expecting 0)")
	}
	n.Node = c
}
