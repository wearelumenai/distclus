package mcmc

import "distclus/core"

// NewAlgo creates a new kmeans algo
func NewAlgo(mcmcConf Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) *core.Algo {
	SetConfigDefaults(&mcmcConf)
	Verify(mcmcConf)
	var distrib = hanfleArgs(args)
	var impl = getImpl(mcmcConf, initializer, data, distrib)
	return buildAlgo(mcmcConf, impl, space)
}

func hanfleArgs(args []interface{}) (distrib Distrib) {
	if len(args) == 1 {
		distrib = args[0].(Distrib)
	}
	return
}

func getImpl(mcmcConf Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) Impl {
	var implFunc func(Conf, core.Initializer, []core.Elemt, Distrib) Impl
	if mcmcConf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(mcmcConf, initializer, data, distrib)
	return impl
}

func buildAlgo(mcmcConf Conf, impl Impl, space core.Space) *core.Algo {
	var conf = core.Conf{ImplConf: mcmcConf}
	var algo = core.NewAlgo(conf, &impl, space)
	return &algo
}
