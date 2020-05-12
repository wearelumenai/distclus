package core

import "github.com/wearelumenai/distclus/figures"

// Impl concrete algorithms
type Impl interface {
	// initialize the algorithm
	Init(OCModel) (Clust, error)
	// process one algorithm iteration
	Iterate(OCModel) (Clust, figures.RuntimeFigures, error)
	// push a data. The second argument is the model
	Push(Elemt, OCModel) error
	// Get a copy of  impl with new conf and space
	Copy(OCModel) (Impl, error)
}
