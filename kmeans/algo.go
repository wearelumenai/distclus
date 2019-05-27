// Package kmeans provides k-means based implementation of online clustering
package kmeans

import "distclus/core"

// NewAlgo creates a new kmeans algo
func NewAlgo(kmeansConf Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) *core.Algo {
	SetConfigDefaults(&kmeansConf)
	Verify(kmeansConf)
	var impl = getImpl(kmeansConf, initializer, data, args)
	return buildAlgo(kmeansConf, impl, space)
}

func buildAlgo(kmeansConf Conf, impl Impl, space core.Space) *core.Algo {
	var conf = core.Conf{ImplConf: kmeansConf}
	var algo = core.NewAlgo(conf, &impl, space)
	return &algo
}

func getImpl(kmeansConf Conf, initializer core.Initializer, data []core.Elemt, args []interface{}) Impl {
	var implFunc func(Conf, core.Initializer, []core.Elemt, ...interface{}) Impl
	if kmeansConf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(kmeansConf, initializer, data, args...)
	return impl
}
