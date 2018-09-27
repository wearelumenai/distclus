package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"time"
)

type MCMC struct {
	*core.AbstractAlgo
	MCMCSupport
	config    MCMCConf
	store     CenterStore
	distrib   MCMCDistrib
	uniform   distuv.Uniform
	iter, acc int
}

type MCMCSupport interface {
	Iterate(core.Clust, int) core.Clust
	Loss(core.Clust) float64
}

func NewSeqMCMC(config MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	setConfigDefaults(&config)
	config.Verify()

	var m MCMC
	m.AbstractAlgo = core.NewAlgo(config.AlgoConf, data, initializer)
	m.AbstractAlgo.RunAlgorithm = m.runAlgorithm
	m.config = config
	m.distrib = distrib
	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.config.RGen}
	m.MCMCSupport = SeqMCMCSupport{buffer: &m.Buffer, config: m.config}
	m.store = NewCenterStore(&m.Buffer, config.Space, m.config.RGen)
	return &m
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

func (mcmc *MCMC) Centroids() (c core.Clust, err error) {
	return mcmc.AbstractAlgo.Centroids()
}

func (mcmc *MCMC) Push(elemt core.Elemt) (err error) {
	return mcmc.AbstractAlgo.Push(elemt)
}

func (mcmc *MCMC) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return mcmc.AbstractAlgo.Predict(elemt, push)
}

func (mcmc *MCMC) Run(async bool) {
	mcmc.AbstractAlgo.Run(async)
}

func (mcmc *MCMC) Close() {
	mcmc.AbstractAlgo.Close()
}

func (mcmc *MCMC) runAlgorithm(closing <-chan bool) {
	var current = proposal{
		k:    mcmc.config.InitK,
		loss: mcmc.Loss(mcmc.Clust),
		pdf:  mcmc.proba(mcmc.Clust, mcmc.Clust),
	}

	for i, loop := 0, true; i < mcmc.config.McmcIter && loop; i++ {
		select {
		case <- closing:
			loop = false

		default:
			current = mcmc.doIter(current)
			mcmc.Apply()
		}
	}
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (mcmc *MCMC) doIter(current proposal) proposal {

	var prop = mcmc.propose(current)

	if mcmc.accept(current, prop) {
		current = prop
		mcmc.Clust = prop.centers
		mcmc.store.SetCenters(mcmc.Clust)
		mcmc.acc += 1
	}

	mcmc.iter += 1
	return current
}

func (mcmc *MCMC) propose(current proposal) proposal {
	var prop proposal
	prop.k = mcmc.nextK(current.k)
	var centers = mcmc.store.GetCenters(prop.k, mcmc.Clust)
	centers = mcmc.Iterate(centers, 1)
	prop.centers = mcmc.alter(centers)
	prop.loss = mcmc.Loss(prop.centers)
	prop.pdf = mcmc.proba(prop.centers, centers)
	return prop
}

func (mcmc *MCMC) accept(current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = mcmc.config.L2B() * float64(mcmc.config.Dim*(current.k-prop.k))
	var rGibbs = mcmc.config.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return mcmc.uniform.Rand() < rho
}

func (mcmc *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(mcmc.config.ProbaK, mcmc.config.RGen)]

	switch {
	case newK < 1:
		return 1
	case newK > mcmc.config.MaxK:
		return mcmc.config.MaxK
	case newK > len(mcmc.Data):
		return len(mcmc.Data)
	default:
		return newK
	}
}

func (mcmc *MCMC) alter(clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))

	for i := range clust {
		result[i] = mcmc.distrib.Sample(clust[i])
	}

	return result
}

func (mcmc *MCMC) proba(x, mu core.Clust) (p float64) {
	p = 0.
	for i := range x {
		p += mcmc.distrib.Pdf(mu[i], x[i])
	}
	return p
}

func (mcmc *MCMC) AcceptRatio() float64 {
	return float64(mcmc.acc) / float64(mcmc.iter)
}
