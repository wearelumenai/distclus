package core

// Impl concrete algorithms
type Impl interface {
	// initialize the algorithm
	Init(OCModel) (Clust, error)
	// process one algorithm iteration
	Iterate(OCModel) (Clust, RuntimeFigures, error)
	// push a data. The second argument is the model
	Push(Elemt, OCModel) error
	// Get a copy of  impl
	Copy(OCModel) (Impl, error)
}
