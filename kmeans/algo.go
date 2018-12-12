package kmeans

import "distclus/core"

// Algo is kmeans algorithm specific structure
type Algo struct {
	*core.Algo
}

// NewAlgo creates a new kmeans algo
func NewAlgo(conf core.Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) Algo {
	var kmeansConf = conf.ImplConf.(Conf)
	var implFunc func(*Conf, core.Initializer, []core.Elemt, ...interface{}) Impl
	if kmeansConf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(&kmeansConf, initializer, data, args...)
	conf.ImplConf = kmeansConf
	var algo = core.NewAlgo(conf, &impl, space)
	return Algo{Algo: &algo}
}
