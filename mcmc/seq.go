package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
)

// NewSeqImpl returns a sequantial mcmc implementation
func NewSeqImpl(conf core.Conf, initializer core.Initializer, data []core.Elemt, args ...interface{}) (impl Impl) {
	var distrib Distrib
	if len(args) == 1 {
		distrib = args[0].(Distrib)
	}
	impl = NewImpl(conf.(Conf), distrib, initializer, data)
	return
}

// SeqStrategy strategy structure
type SeqStrategy struct {
}

// Iterate execute the algorithm
func (strategy *SeqStrategy) Iterate(conf Conf, space core.Space, clust core.Clust, buffer core.Buffer, iter int) (result core.Clust) {
	var kmeansConf = kmeans.Conf{
		K:    len(clust),
		Iter: iter,
		RGen: conf.RGen,
	}
	var impl = kmeans.NewSeqImpl(kmeansConf, clust.Initializer, buffer.Data())
	var algo = core.NewAlgo(kmeansConf, &impl, space)
	algo.Run(false)
	algo.Close()
	result, _ = algo.Centroids()

	return
}

// Loss calculates input centroids
func (strategy *SeqStrategy) Loss(conf Conf, space core.Space, proposal core.Clust, buffer core.Buffer) float64 {
	return proposal.Loss(buffer.Data(), space, conf.Norm)
}
