// Package mcmc provides MCMC based implementation of online clustering (cf https://hal.inria.fr/hal-01264233).
package mcmc

import "github.com/wearelumenai/distclus/v0/core"

// NewAlgo creates a new kmeans algo
func NewAlgo(conf Conf, space core.Space, data []core.Elemt, initializer core.Initializer, distrib Distrib) *core.Algo {
	conf.Verify()
	var impl = getImpl(conf, initializer, data, distrib)
	return core.NewAlgo(&conf, impl, space)
}

func getImpl(conf Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) *Impl {
	var implFunc func(Conf, core.Initializer, []core.Elemt, Distrib) Impl
	if conf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(conf, initializer, data, distrib)
	return &impl
}
