package algebrain

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
)

// A SampleList wraps a slice of Samples for training.
type SampleList []*Sample

// Len returns the number of samples.
func (s SampleList) Len() int {
	return len(s)
}

// Swap swaps two samples.
func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Slice returns a subset of the sample list.
func (s SampleList) Slice(i, j int) anysgd.SampleList {
	return append(SampleList{}, s[i:j]...)
}

// Batch is a batch of fetched training samples.
type Batch struct {
	EncIn  anyseq.Seq
	DecIn  anyseq.Seq
	DecOut anyseq.Seq
}

// A Trainer computes costs and gradients for a Network.
type Trainer struct {
	Network *Network

	// LastCost is set by every call to Gradient.
	LastCost anyvec.Numeric
}

// Fetch creates a *Batch from a SampleList.
func (t *Trainer) Fetch(s anysgd.SampleList) anysgd.Batch {
	var encIn, decIn, decOut [][]anyvec.Vector
	for i := 0; i < s.Len(); i++ {
		sample := s.(SampleList)[i]
		encIn = append(encIn, sample.InputSequence())
		decIn = append(decIn, sample.DecoderInSequence())
		decOut = append(decOut, sample.DecoderOutSequence())
	}
	return &Batch{
		EncIn:  anyseq.ConstSeqList(encIn),
		DecIn:  anyseq.ConstSeqList(decIn),
		DecOut: anyseq.ConstSeqList(decOut),
	}
}

// TotalCost computes the cost for the *Batch.
func (t *Trainer) TotalCost(b anysgd.Batch) anydiff.Res {
	trainer, batch := t.tempTrainer(b)
	return trainer.TotalCost(batch)
}

// Gradient computes the cost gradient.
// It sets t.LastCost to the cost.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	trainer, batch := t.tempTrainer(b)
	res := trainer.Gradient(batch)
	t.LastCost = trainer.LastCost
	return res
}

func (t *Trainer) tempTrainer(b anysgd.Batch) (*anys2s.Trainer, *anys2s.Batch) {
	return &anys2s.Trainer{
			Func: func(s anyseq.Seq) anyseq.Seq {
				decOut := b.(*Batch).DecIn
				enc := t.Network.Encoder.Apply(decOut)
				return anyseq.Pool(enc, func(enc anyseq.Seq) anyseq.Seq {
					block := t.Network.Align.Block(enc)
					return anyrnn.Map(enc, block)
				})
			},
			Cost:    anynet.DotCost{},
			Params:  t.Network.Parameters(),
			Average: true,
		}, &anys2s.Batch{
			Inputs:  b.(*Batch).EncIn,
			Outputs: b.(*Batch).DecOut,
		}
}
