package algebrain

import "github.com/unixpickle/sgd"

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

// GetSample gives the *Sample at the given index.
func (s SampleSet) GetSample(i int) interface{} {
	return s[i]
}

// Copy returns a copy of the sample set.
func (s SampleSet) Copy() sgd.SampleSet {
	return append(SampleSet{}, s...)
}

// Subset returns a subset of the sample set.
func (s SampleSet) Subset(i, j int) sgd.SampleSet {
	return s[i:j]
}
