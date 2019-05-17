package streaming

import "distclus/core"

func NewAlgo(conf Conf, space core.Space, data []core.Elemt, args ...interface{}) *core.Algo {
	SetConfigDefaults(&conf)
	Verify(conf)
	if conf.BufferSize < len(data) {
		panic("buffer size must be greater than initial data")
	}
	var impl = getImpl(conf, data)
	return buildAlgo(conf, impl, space)
}

func getImpl(strConf Conf, elemts []core.Elemt) Impl {
	return NewImpl(strConf, elemts)
}

func buildAlgo(strConf Conf, impl Impl, space core.Space) *core.Algo {
	var conf = core.Conf{ImplConf: strConf}
	var algo = core.NewAlgo(conf, &impl, space)
	return &algo
}
