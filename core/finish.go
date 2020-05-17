package core

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
func IsFinished(finishing Finishing, model OCModel) bool {
	return finishing != nil && finishing.IsFinished(model)
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

// IterFinishing is a finishing with iterations such as parameter
type IterFinishing struct {
	Iter        int
	IterPerData int
}

// IsFinished is Iterations Convergence implementation
func (i IterFinishing) IsFinished(ocm OCModel) (cond bool) {
	if i.Iter > 0 || i.IterPerData > 0 {
		var rf = ocm.RuntimeFigures()
		var iterations = rf[Iterations]
		var pushedData = rf[PushedData]

		cond = int(iterations) >= (i.Iter + int(pushedData)*i.IterPerData)
	}
	return
}

// NewIterFinishing returns new instance
func NewIterFinishing(iter int, iterPerData int) IterFinishing {
	return IterFinishing{
		Iter:        iter,
		IterPerData: iterPerData,
	}
}

// StatusFinishing compare OCModel status
type StatusFinishing struct {
	Status []ClustStatus // Specific status finish condition
	Error  bool          // if true and ocm failed, finish
}

// IsFinished StatusFinishing finish condition
func (sf StatusFinishing) IsFinished(ocm OCModel) (cond bool) {
	var status = ocm.Status()
	cond = sf.Error && status.Error != nil
	if !cond && len(sf.Status) > 0 {
		for _, sfStatus := range sf.Status {
			if cond = (sfStatus == status.Value); cond {
				break
			}
		}
	}
	return
}

// NewStatusFinishing returns new instance
func NewStatusFinishing(error bool, status ...ClustStatus) StatusFinishing {
	return StatusFinishing{
		Status: status,
		Error:  error,
	}
}
