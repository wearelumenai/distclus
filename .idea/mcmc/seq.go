package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"time"

	"golang.org/x/exp/rand"
)

func NewSeqMCMC(config MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	return NewMCMC(config, distrib, initializer, data)
}

type SeqMCMCStrategy struct {
	Config MCMCConf
	Buffer *core.DataBuffer
}

func setConfigDefaults(conf *MCMCConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if len(conf.ProbaK) == 0 {
		conf.ProbaK = []float64{1, 0, 9}
	}
	if conf.MaxK == 0 {
		conf.MaxK = 16
	}
}

func (strategy SeqMCMCStrategy) Iterate(clust core.Clust, iter int) (result core.Clust) {
	var conf = kmeans.KMeansConf{
		AlgorithmConf: core.AlgorithmConf{
			Space: strategy.Config.Space,
		},
		K:    len(clust),
		Iter: iter,
		RGen: strategy.Config.RGen,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()
	result, _ = km.Centroids()

	return
}

func (strategy SeqMCMCStrategy) Loss(proposal core.Clust) float64 {
	return proposal.Loss(strategy.Buffer.Data, strategy.Config.Space, strategy.Config.Norm)
}
