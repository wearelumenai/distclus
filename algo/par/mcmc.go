package par

import (
	"distclus/algo"
	"distclus/core"
)

type MCMCConf struct {
	algo.MCMCConf
}

type MCMC struct {
	seq algo.MCMC
	config MCMCConf
}

func NewMCMC(conf MCMCConf, distrib algo.MCMCDistrib) MCMC  {
	var mcmc MCMC

	mcmc.seq = algo.NewMCMC(conf.MCMCConf, distrib)
	mcmc.config = conf

	return mcmc
}

func (mcmc *MCMC) Push(elemt core.Elemt) {
	mcmc.seq.Push(elemt)
}

func (mcmc *MCMC) Centroids() (algo.Clust, error) {
	return mcmc.seq.Centroids()
}

func (*MCMC) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	panic("implement me")
}

// Compute loss proposal based on Clust.Loss
func (m *MCMC) loss(proposal algo.Clust) float64 {
	return proposal.Loss(m.seq.Data, m.config.Space, m.config.Norm)
}

// Make an iterate for a proposal running with kmeans
func (m *MCMC) iterate(k int, clust *algo.Clust) {
	var km = algo.NewKMeans(k, 1, m.config.Space, clust.Initializer)

	km.Data = m.seq.Data

	km.Run()
	km.Close()

	var c, _ = km.Centroids()
	clust = &c
}

func (mcmc *MCMC) Run() {
	algo.MCMCLoop(&mcmc.seq, mcmc.iterate, mcmc.loss)
}

func (mcmc *MCMC) Close() {
	mcmc.seq.Close()
}

