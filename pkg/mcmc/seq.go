package mcmc

import (
	"lumenai.fr/v0/distclus/pkg/core"
	"lumenai.fr/v0/distclus/pkg/kmeans"

	"gonum.org/v1/gonum/stat/distuv"
)

// NewSeqImpl returns a sequantial mcmc implementation
func NewSeqImpl(conf Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) Impl {
	return Impl{
		buffer:      core.NewDataBuffer(data, conf.FrameSize),
		initializer: initializer,
		uniform:     distuv.Uniform{Max: 1, Min: 0, Src: conf.RGen},
		store:       NewCenterStore(conf.RGen),
		strategy:    &SeqStrategy{},
		distrib:     distrib,
	}
}

// SeqStrategy strategy structure
type SeqStrategy struct {
}

// Iterate execute the algorithm
func (strategy *SeqStrategy) Iterate(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt, iter int) (result core.Clust) {
	var kmeansConf = kmeans.Conf{
		K:    len(centroids),
		RGen: conf.RGen,
		Conf: core.Conf{
			Iter: iter,
		},
	}
	var algo = kmeans.NewAlgo(kmeansConf, space, data, centroids.Initializer)
	algo.Batch(0, 0)
	result, _ = algo.Centroids()

	return
}

// Loss calculates loss for the given proposal and data
func (strategy *SeqStrategy) Loss(conf Conf, space core.Space, proposal core.Clust, data []core.Elemt) float64 {
	return proposal.TotalLoss(data, space, conf.Norm)
}
