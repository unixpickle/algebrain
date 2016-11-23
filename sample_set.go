package algebrain

import (
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

// A SampleSet wraps a slice of Samples in an
// sgd.SampleSet.
type SampleSet []*Sample

// Len returns the number of samples.
func (s SampleSet) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// GetSample produces a seqtoseq.Sample for the sample.
func (s SampleSet) GetSample(i int) interface{} {
	sample := s[i]

	res := seqtoseq.Sample{}
	for _, x := range sample.Query {
		res.Inputs = append(res.Inputs, oneHotVector(x))
		res.Outputs = append(res.Outputs, zeroVector())
	}
	res.Inputs = append(res.Inputs, oneHotVector(Terminator))
	res.Outputs = append(res.Outputs, zeroVector())

	var last rune = Terminator
	for _, x := range sample.Response {
		res.Inputs = append(res.Inputs, oneHotVector(last))
		res.Outputs = append(res.Outputs, oneHotVector(x))
		last = x
	}

	res.Inputs = append(res.Inputs, oneHotVector(last))
	res.Outputs = append(res.Outputs, oneHotVector(Terminator))

	return res
}

// Copy returns a copy of the sample set.
func (s SampleSet) Copy() sgd.SampleSet {
	return append(SampleSet{}, s...)
}

// Subset returns a subset of the sample set.
func (s SampleSet) Subset(i, j int) sgd.SampleSet {
	return s[i:j]
}
