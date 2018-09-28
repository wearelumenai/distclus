package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"time"
)

func NewSeqMCMC(config MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	setConfigDefaults(&config)
	config.Verify()

	var mcmc MCMC
	var algoTemplateMethods = core.AlgorithmTemplateMethods{
		Initialize: mcmc.initializeAlgorithm,
		Run:        mcmc.runAlgorithm,
	}
	mcmc.template = core.NewAlgo(config.AlgorithmConf, data, algoTemplateMethods)

	mcmc.config = config
	mcmc.initializer = initializer
	mcmc.distrib = distrib
	mcmc.uniform = distuv.Uniform{Max: 1, Min: 0, Src: mcmc.config.RGen}
	mcmc.store = NewCenterStore(&mcmc.template.Buffer, config.Space, mcmc.config.RGen)
	mcmc.strategy = SeqMCMCStrategy{Buffer: &mcmc.template.Buffer, Config: mcmc.config}

	return &mcmc
}

type SeqMCMCStrategy struct {
	Config MCMCConf
	Buffer *core.Buffer
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

func (strategy SeqMCMCStrategy) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{
		AlgorithmConf: core.AlgorithmConf{
			InitK: len(clust),
			Space: strategy.Config.Space,
			RGen: strategy.Config.RGen,
		},
		Iter: iter,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()
	var result, _ = km.Centroids()

	return result
}

func (strategy SeqMCMCStrategy) Loss(proposal core.Clust) float64 {
	return proposal.Loss(strategy.Buffer.Data, strategy.Config.Space, strategy.Config.Norm)
}
