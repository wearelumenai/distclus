package core

import "go.lumenai.fr/distclus/v0/figures"

// Impl concrete algorithms
type Impl interface {
	// initialize the algorithm
	Init(ImplConf, Space, Clust) (Clust, error)
	// process one algorithm iteration
	Iterate(ImplConf, Space, Clust) (Clust, figures.RuntimeFigures, error)
	// push a data. The second argument is true if algo is running
	Push(Elemt, bool) error
	// Get a copy of  impl with new conf and space
	Copy(ImplConf, Space) (Impl, error)
}

// ImplConf is implementation configuration interface
type ImplConf interface {
	Verify()
	AlgoConf() *Conf
}
