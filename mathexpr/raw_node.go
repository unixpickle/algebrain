package mathexpr

// A RawNode represents a raw token such as a variable
// name or a constant value.
type RawNode string

// Precedence returns AtomicPrecedence.
func (n RawNode) Precedence() Precedence {
	return AtomicPrecedence
}

// String returns the raw string.
func (n RawNode) String() string {
	return string(n)
}

// Children returns the empty slice.
func (n RawNode) Children() []Node {
	return nil
}

// SetChild panics since there are no children.
func (n RawNode) SetChild(idx int, c Node) {
	panic("no children")
}
