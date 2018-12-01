package mcmc

import "distclus/core"

// Algo is a mcmc algorithm
type Algo struct {
	*core.Algo
}

// NewAlgo creates a new kmeans algo
func NewAlgo(conf core.Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) Algo {
	var mcmcConf = conf.(Conf)
	var distrib Distrib
	if len(args) == 1 {
		distrib = args[0].(Distrib)
	}
	var implFunc func(*Conf, core.Initializer, []core.Elemt, Distrib) Impl
	if mcmcConf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(&mcmcConf, initializer, data, distrib)
	var algo = core.NewAlgo(mcmcConf, &impl, space)
	return Algo{Algo: &algo}
}

// AcceptRatio returns ratio between acc and iter
func (algo *Algo) AcceptRatio() float64 {
	return algo.Algo.Impl.(*Impl).AcceptRatio()
}
