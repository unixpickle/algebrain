package mathexpr

import "testing"

func TestStrings(t *testing.T) {
	exprs := []Node{
		&BinaryOp{Left: RawNode("2"), Right: RawNode("3"), Op: "*"},
		&BinaryOp{
			Left:  &BinaryOp{Left: RawNode("2"), Right: RawNode("3"), Op: "*"},
			Right: &BinaryOp{Left: RawNode("3"), Right: RawNode("2"), Op: "/"},
			Op:    "+",
		},
		&BinaryOp{
			Left:  &BinaryOp{Left: RawNode("2"), Right: RawNode("3"), Op: "-"},
			Right: &BinaryOp{Left: RawNode("3"), Right: RawNode("2"), Op: "*"},
			Op:    "/",
		},
		&BinaryOp{
			Left: &BinaryOp{
				Left:  &BinaryOp{Left: RawNode("2"), Right: RawNode("3"), Op: "-"},
				Right: &BinaryOp{Left: RawNode("3"), Right: RawNode("2"), Op: "*"},
				Op:    "+",
			},
			Right: &FuncOp{
				Name: "P",
				Args: []Node{
					RawNode("3"),
					&BinaryOp{
						Left:  RawNode("2"),
						Right: RawNode("x"),
						Op:    "^",
					},
				},
			},
			Op: "^",
		},
	}
	strs := []string{
		"2*3",
		"2*3+3/2",
		"(2-3)/(3*2)",
		"((2-3)+3*2)^P(3, 2^x)",
	}
	for i, x := range exprs {
		actual := x.String()
		expected := strs[i]
		if actual != expected {
			t.Errorf("expr %d: expected %s got %s", i, expected, actual)
		}
	}
}
