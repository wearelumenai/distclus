package template // to rename for specific algorithm

import "lumenai.fr/v0/distclus/pkg/core"

// NewAlgo creates a new algorithm with a specific implementation
func NewAlgo(conf Conf, space core.Space, data []core.Elemt, args ...interface{}) *core.Algo {
	var impl = NewImpl(conf, data)
	return core.NewAlgo(&conf, &impl, space)
}
