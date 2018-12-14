package kmeans

import (
	"distclus/core"
)

// NewSeqImpl returns a sequential algorithm execution
func NewSeqImpl(conf *Conf, initializer core.Initializer, data []core.Elemt, args ...interface{}) Impl {
	return NewImpl(conf, initializer, data)
}

// SeqStrategy defines strategy for sequential execution
type SeqStrategy struct {
}

// Iterate processes input cluster
func (strategy *SeqStrategy) Iterate(space core.Space, centroids core.Clust, data []core.Elemt) core.Clust {
	var result, _ = centroids.AssignDBA(data, space)
	return strategy.buildResult(centroids, result)
}

func (strategy SeqStrategy) buildResult(centroids core.Clust, result core.Clust) core.Clust {
	for i := 0; i < len(result); i++ {
		if result[i] == nil {
			result[i] = centroids[i]
		}
	}
	return result
}
