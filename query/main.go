package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/unixpickle/algebrain"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func main() {
	if len(os.Args) != 2 {
		essentials.Die("Usage:", os.Args[0], "<net_file>")
	}
	var net *algebrain.Network
	if err := serializer.LoadAny(os.Args[1], &net); err != nil {
		essentials.Die("Failed to load block:", err)
	}
	for {
		fmt.Println(net.Query(readLine()))
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
