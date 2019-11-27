package streaming

import "distclus/core"

// NewAlgo creates a new algorithm with a streaming implementation
func NewAlgo(conf Conf, space core.Space, data []core.Elemt) *core.Algo {
	conf.Verify()
	if conf.BufferSize < len(data) {
		panic("buffer size must be greater than initial data")
	}
	var impl = getImpl(conf, data)
	return core.NewAlgo(&conf, &impl, space)
}

func getImpl(strConf Conf, elemts []core.Elemt) Impl {
	return NewImpl(strConf, elemts)
}
