// Package kmeans provides k-means based implementation of online clustering
package kmeans

import "distclus/core"

// NewAlgo creates a new kmeans algo
func NewAlgo(conf Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) *core.Algo {
	conf.Verify()
	var impl = getImpl(conf, initializer, data, args)
	return buildAlgo(conf, impl, space)
}

func buildAlgo(conf Conf, impl Impl, space core.Space) (algo *core.Algo) {
	return core.NewAlgo(&conf, &impl, space)
}

func getImpl(conf Conf, initializer core.Initializer, data []core.Elemt, args []interface{}) Impl {
	var implFunc func(Conf, core.Initializer, []core.Elemt, ...interface{}) Impl
	if conf.Par {
		implFunc = NewParImpl
	} else {
		implFunc = NewSeqImpl
	}
	var impl = implFunc(conf, initializer, data, args...)
	return impl
}
