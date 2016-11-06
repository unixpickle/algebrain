package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/unixpickle/algebrain"
)

func main() {
	if len(os.Args) != 2 {
		die("Usage:", os.Args[0], "<net_file>")
	}
	block, err := algebrain.LoadBlock(os.Args[1])
	if err != nil {
		die("Load block:", err)
	}
	for {
		fmt.Println(block.Query(readLine()))
	}
}

func readLine() string {
	fmt.Print("Query> ")
	var res bytes.Buffer
	for {
		var x [1]byte
		if n, err := os.Stdin.Read(x[:]); err != nil {
			panic(err)
		} else if n != 0 {
			if x[0] == '\n' {
				break
			} else if x[0] != '\r' {
				res.WriteByte(x[0])
			}
		}
	}
	return res.String()
}

func die(args ...interface{}) {
	fmt.Fprintln(os.Stderr, args...)
	os.Exit(1)
}
