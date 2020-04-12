package kmeans

import (
	"github.com/wearelumenai/distclus/core"
)

// NewParImpl parallelizes algorithm implementation
func NewParImpl(conf Conf, initializer core.Initializer, data []core.Elemt, args ...interface{}) (impl Impl) {
	impl = NewSeqImpl(conf, initializer, data)
	impl.strategy = ParStrategy{
		Degree: conf.NumCPU,
	}
	return
}

// ParStrategy parallelizes algorithm strategy
type ParStrategy struct {
	Degree int
}

// Iterate processes input cluster
func (strategy ParStrategy) Iterate(space core.Space, centroids core.Clust, data []core.Elemt) core.Clust {
	result, _ := centroids.ParReduceDBA(data, space, strategy.Degree)
	return result
}
