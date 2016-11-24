package algebrain

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const attentorBatchSize = 16

// An attentor implements the attention mechanism as a
// neuralstruct.RStruct, although the r-operator methods
// are not implemented.
//
// Note that this is not what neuralstruct is for.
// Using neuralstruct for this is a complete, utter hack.
type attentor struct {
	Encoded seqfunc.Result
	Network neuralnet.Network
	Grad    autofunc.Gradient
}

// ControlSize returns the vector size for the focus
// request.
func (a *attentor) ControlSize() int {
	return focusInfoSize
}

// DataSize returns the size of the result of focusing.
func (a *attentor) DataSize() int {
	return encoderOutSize
}

// StartState returns a state corresponding to no focus.
func (a *attentor) StartState() neuralstruct.State {
	return &attentorState{
		attentor: a,
		output:   &autofunc.Variable{Vector: make(linalg.Vector, a.DataSize())},
	}
}

// StartRState panics.
func (a *attentor) StartRState() neuralstruct.RState {
	panic("not implemented")
}

type attentorState struct {
	attentor *attentor
	output   autofunc.Result
	ctrl     *autofunc.Variable
}

func (a *attentorState) Data() linalg.Vector {
	return a.output.Output()
}

func (a *attentorState) Gradient(us linalg.Vector, _ neuralstruct.Grad) (ctrlGrad linalg.Vector,
	down neuralstruct.Grad) {
	ctrlGrad = make(linalg.Vector, len(a.ctrl.Vector))
	if a.ctrl != nil {
		a.attentor.Grad[a.ctrl] = ctrlGrad
	}
	a.output.PropagateGradient(us, a.attentor.Grad)
	if a.ctrl != nil {
		delete(a.attentor.Grad, a.ctrl)
	}
	return
}

func (a *attentorState) NextState(ctrl linalg.Vector) neuralstruct.State {
	ctrlPool := &autofunc.Variable{Vector: ctrl}
	ctrlAugmented := seqfunc.Map(a.attentor.Encoded, func(in autofunc.Result) autofunc.Result {
		return autofunc.Concat(ctrlPool, in)
	})
	mapper := seqfunc.FixedMapBatcher{
		B:         a.attentor.Network.BatchLearner(),
		BatchSize: attentorBatchSize,
	}
	energies := mapper.ApplySeqs(ctrlAugmented)
	exps := seqfunc.Map(energies, autofunc.Exp{}.Apply)
	mag := autofunc.Inverse(seqfunc.AddAll(exps))
	probs := seqfunc.Map(exps, func(in autofunc.Result) autofunc.Result {
		return autofunc.Mul(mag, in)
	})
	scaled := seqfunc.MapN(func(ins ...autofunc.Result) autofunc.Result {
		return autofunc.ScaleFirst(ins[0], ins[1])
	}, a.attentor.Encoded, probs)
	return &attentorState{
		attentor: a.attentor,
		output:   seqfunc.AddAll(scaled),
		ctrl:     ctrlPool,
	}
}
