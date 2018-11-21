package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

type MCMC struct {
	template    *core.AlgorithmTemplate
	data        *core.DataBuffer
	config      MCMCConf
	strategy    MCMCStrategy
	initializer core.Initializer
	uniform     distuv.Uniform
	distrib     MCMCDistrib
	store       CenterStore
	iter, acc   int
}

type MCMCStrategy interface {
	Iterate(core.Clust, int) core.Clust
	Loss(core.Clust) float64
}

func (mcmc *MCMC) Centroids() (c core.Clust, err error) {
	return mcmc.template.Centroids()
}

func (mcmc *MCMC) Push(elemt core.Elemt) (err error) {
	return mcmc.template.Push(elemt)
}

func (mcmc *MCMC) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return mcmc.template.Predict(elemt, push)
}

func (mcmc *MCMC) Run(async bool) {
	mcmc.template.Run(async)
}

func (mcmc *MCMC) Close() {
	mcmc.template.Close()
}

func (mcmc *MCMC) initializeAlgorithm() (centroids core.Clust, ready bool) {
	mcmc.data.Apply()
	return mcmc.initializer(mcmc.config.InitK, mcmc.data.Data, mcmc.config.Space, mcmc.config.RGen)
}

func NewMCMC(config MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) (mcmc *MCMC) {
	setConfigDefaults(&config)
	config.Verify()
	mcmc = &MCMC{
		data:        core.NewDataBuffer(data, config.FrameSize),
		config:      config,
		initializer: initializer,
		distrib:     distrib,
	}
	mcmc.uniform = distuv.Uniform{Max: 1, Min: 0, Src: mcmc.config.RGen}
	mcmc.store = NewCenterStore(mcmc.data, config.Space, mcmc.config.RGen)
	mcmc.strategy = SeqMCMCStrategy{Buffer: mcmc.data, Config: mcmc.config}

	var algoTemplateMethods = core.AlgorithmTemplateMethods{
		Initialize: mcmc.initializeAlgorithm,
		Run:        mcmc.runAlgorithm,
	}
	mcmc.template = core.NewAlgorithmTemplate(config.AlgorithmConf, mcmc.data, algoTemplateMethods)

	return
}

func (mcmc *MCMC) runAlgorithm(closing <-chan bool) {
	var current = proposal{
		k:       mcmc.config.InitK,
		centers: mcmc.template.Clust,
		loss:    mcmc.strategy.Loss(mcmc.template.Clust),
		pdf:     mcmc.proba(mcmc.template.Clust, mcmc.template.Clust),
	}

	for i, loop := 0, true; i < mcmc.config.McmcIter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			current = mcmc.doIter(current)
			mcmc.data.Apply()
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
		mcmc.template.Clust = prop.centers
		mcmc.store.SetCenters(mcmc.template.Clust)
		mcmc.acc += 1
	}

	mcmc.iter += 1
	return current
}

func (mcmc *MCMC) propose(current proposal) proposal {
	var prop proposal
	prop.k = mcmc.nextK(current.k)
	var centers = mcmc.store.GetCenters(prop.k, mcmc.template.Clust)
	centers = mcmc.strategy.Iterate(centers, 1)
	prop.centers = mcmc.alter(centers)
	prop.loss = mcmc.strategy.Loss(prop.centers)
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
	case newK > len(mcmc.data.Data):
		return len(mcmc.data.Data)
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
