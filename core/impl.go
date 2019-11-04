package core

import "distclus/figures"

// Impl concrete algorithms
type Impl interface {
	// initialize the algorithm
	Init(ImplConf, Space, Clust) (Clust, error)
	// process one algorithm iteration
	Iterate(ImplConf, Space, Clust) (Clust, figures.RuntimeFigures, error)
	// push a data
	Push(Elemt) error
	// set algorithm to online clustering mode
	SetOC() error
}

// ImplConf is implementation configuration interface
type ImplConf interface {
	Verify()
	AlgoConf() *Conf
}
