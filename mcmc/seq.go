package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
)

// NewSeqImpl returns a sequantial mcmc implementation
func NewSeqImpl(conf core.Conf, distrib Distrib, initializer core.Initializer, data []core.Elemt) Impl {
	return NewImpl(conf.(Conf), distrib, initializer, data)
}

// SeqStrategy strategy structure
type SeqStrategy struct {
}

// Iterate execute the algorithm
func (strategy *SeqStrategy) Iterate(clust core.Clust, iter int) (result core.Clust) {
	var conf = kmeans.Conf{
		K:    len(clust),
		Iter: iter,
		RGen: Conf.RGen,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()
	result, _ = km.Centroids()

	return
}

// Loss calculates input centroids
func (strategy *SeqStrategy) Loss(conf Conf, space core.Space, proposal core.Clust) float64 {
	return proposal.Loss(strategy.Buffer.Data, space, conf.Norm)
}
