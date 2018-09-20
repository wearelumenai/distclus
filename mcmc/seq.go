package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
)

type SeqMCMCSupport struct {
	config MCMCConf
	buffer *core.Buffer
}

func (support SeqMCMCSupport) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{
		K: len(clust),
		Iter: iter,
		Space: support.config.Space,
		RGen: support.config.RGen,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, support.buffer.Data)

	km.Run(false)
	km.Close()
	var result, _ = km.Centroids()

	return result
}

func (support SeqMCMCSupport) Loss(proposal core.Clust) float64 {
	return proposal.Loss(support.buffer.Data, support.config.Space, support.config.Norm)
}
