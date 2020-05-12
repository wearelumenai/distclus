package core

import "github.com/wearelumenai/distclus/figures"

// Finishing defines
type Finishing interface {
	IsFinished(occ OCModel) bool
}

// And applies and rule over finishings
type And struct {
	Finishings []Finishing
}

// Or applies or rule over finishings
type Or struct {
	Finishings []Finishing
}

// NewAnd returns And
func NewAnd(finishings ...Finishing) Finishing {
	return And{Finishings: finishings}
}

// NewOr returns Or
func NewOr(finishings ...Finishing) Finishing {
	return Or{Finishings: finishings}
}

// IsFinished input finishing on input OCModel
func IsFinished(finishing Finishing, model OCModel) (cond bool) {
	defer func() {
		var r = recover()
		if r != nil {
			cond = false
		}
	}()
	cond = finishing != nil && finishing.IsFinished(model)
	return
}

// IsFinished is the And Convergence implementation
func (and And) IsFinished(ocm OCModel) (cond bool) {
	for _, finishing := range and.Finishings {
		cond = IsFinished(finishing, ocm)
		if !cond {
			break
		}
	}
	return
}

// IsFinished is the Or Convergence implementation
func (or Or) IsFinished(ocm OCModel) (cond bool) {
	for _, finishing := range or.Finishings {
		cond = IsFinished(finishing, ocm)
		if cond {
			break
		}
	}
	return
}

// IterationsFinishing is a finishing with iterations such as parameter
type IterationsFinishing struct {
	MaxIter int
}

// IsFinished is Iterations Convergence implementation
func (iterations IterationsFinishing) IsFinished(ocm OCModel) (cond bool) {
	if iterations.MaxIter > 0 {
		var runtimeFigures figures.RuntimeFigures
		runtimeFigures = ocm.RuntimeFigures()
		var lastIterations = runtimeFigures[figures.LastIterations]
		cond = int(lastIterations) >= iterations.MaxIter
	}
	return
}
