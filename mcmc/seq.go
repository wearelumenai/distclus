package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
)

// NewSeqImpl returns a sequantial mcmc implementation
func NewSeqImpl(conf *Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) Impl {
	return NewImpl(conf, initializer, data, distrib)
}

// SeqStrategy strategy structure
type SeqStrategy struct {
}

// Iterate execute the algorithm
func (strategy *SeqStrategy) Iterate(conf Conf, space core.Space, centroids core.Clust, buffer core.Buffer, iter int) (result core.Clust) {
	var kmeansConf = core.Conf{
		ImplConf: kmeans.Conf{
			K:    len(centroids),
			Iter: iter,
			RGen: conf.RGen,
		},
		SpaceConf: nil,
	}
	var algo = kmeans.NewAlgo(kmeansConf, space, buffer.Data(), centroids.Initializer)
	algo.Run(false)
	algo.Close()
	result, _ = algo.Centroids()

	return
}

// Loss calculates input centroids
func (strategy *SeqStrategy) Loss(conf Conf, space core.Space, proposal core.Clust, buffer core.Buffer) float64 {
	return proposal.Loss(buffer.Data(), space, conf.Norm)
}
