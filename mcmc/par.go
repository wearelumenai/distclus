package mcmc

import (
	"go.lumenai.fr/distclus/v0/core"
	"go.lumenai.fr/distclus/v0/kmeans"
)

// NewParImpl returns a new parallelized algorithm implementation
func NewParImpl(conf Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) (impl Impl) {
	impl = NewSeqImpl(conf, initializer, data, distrib)
	impl.strategy = &ParStrategy{
		Degree: conf.NumCPU,
	}
	return
}

// ParStrategy defines a parallelized strategy
type ParStrategy struct {
	Degree int
}

// Iterate is the iterative execution
func (strategy *ParStrategy) Iterate(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt, iter int) core.Clust {
	var kmeansConf = kmeans.Conf{
		Par: true,
		K:   len(centroids),
		Conf: core.Conf{
			Iter: iter,
		},
	}
	var algo = kmeans.NewAlgo(kmeansConf, space, data, centroids.Initializer)

	algo.Batch(0, 0)

	var result, _ = algo.Centroids()

	return result
}

// Loss calculates loss for the given proposal and data in parallel
func (strategy *ParStrategy) Loss(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt) float64 {
	return centroids.ParTotalLoss(data, space, conf.Norm, strategy.Degree)
}
