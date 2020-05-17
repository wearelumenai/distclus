package mcmc

import (
	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/kmeans"
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
func (strategy *ParStrategy) Iterate(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt, iter int) (result core.Clust) {
	var kmeansConf = kmeans.Conf{
		Par: true,
		K:   len(centroids),
		CtrlConf: core.CtrlConf{
			Iter: iter,
		},
	}
	core.PrepareConf(&kmeansConf)
	var model = core.NewSimpleOCModel(&kmeansConf, space, core.NewOCStatus(core.Created), core.RuntimeFigures{}, result)
	var impl = kmeans.NewParImpl(kmeansConf, centroids.Initializer, data)
	result, _ = impl.Init(model)
	for i := 0; i < iter; i++ {
		model = core.NewSimpleOCModel(&kmeansConf, space, core.NewOCStatus(core.Created), core.RuntimeFigures{}, result)
		result, _, _ = impl.Iterate(model)
	}

	return
}

// Loss calculates loss for the given proposal and data in parallel
func (strategy *ParStrategy) Loss(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt) float64 {
	return centroids.ParTotalLoss(data, space, conf.Norm, strategy.Degree)
}
