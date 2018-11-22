package kmeans

import (
	"distclus/core"
)

// NewSeqImpl returns a sequential algorithm execution
func NewSeqImpl(conf core.Conf, initializer core.Initializer, data []core.Elemt, _ ...interface{}) Impl {
	return NewImpl(conf.(Conf), initializer, data)
}

// SeqStrategy defines strategy for sequential execution
type SeqStrategy struct {
}

// Iterate processes input cluster
func (strategy *SeqStrategy) Iterate(space core.Space, centroids core.Clust, buffer core.DataBuffer) core.Clust {
	var result, _ = centroids.AssignDBA(buffer.Data, space)
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
