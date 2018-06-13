package par

import (
	"distclus/algo"
)

type ParMCMCSupport struct {
	config algo.MCMCConf
}

func (supp ParMCMCSupport) Iterate(m algo.MCMC, k int) algo.Clust {
	var clust, _ = m.Centroids()
	var km = NewKMeans(k, 1, supp.config.Space, clust.Initializer)

	km.Data = m.Data

	km.Run()
	km.Close()

	var result, _ = km.Centroids()
	return result

}

func (supp ParMCMCSupport) Loss(m algo.MCMC, proposal algo.Clust) float64 {
	return proposal.Loss(m.Data, supp.config.Space, supp.config.Norm)
}

func NewMCMC(conf algo.MCMCConf, distrib algo.MCMCDistrib) algo.MCMC  {
	var mcmc = algo.NewMCMC(conf, distrib)

	mcmc.MCMCSupport = ParMCMCSupport{conf}

	return mcmc
}

