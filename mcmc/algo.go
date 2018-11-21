package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Impl of MCMC
type Impl struct {
	centroids core.Cluster
	buffer    core.Buffer
	strategy  Strategy
	uniform   distuv.Uniform
	distrib   Distrib
	store     CenterStore
	iter, acc int
}

// Strategy specifies strategy methods
type Strategy interface {
	Iterate(core.Clust, int) core.Clust
	Loss(Conf, core.Space, core.Clust) float64
}

// Init initializes the algorithm
func (impl *Impl) Init(conf core.Conf) (core.Clust, bool) {
	var mcmcConf = conf.(Conf)
	SetConfigDefaults(&mcmcConf)
	Verify(mcmcConf)
	impl.buffer.Apply()
	return impl.initializer(mcmcConf.InitK, impl.buffer.Data, mcmcConf.Space, mcmcConf.RGen)
}

// NewImpl function
func NewImpl(conf Conf, space core.Space, distrib Distrib, initializer core.Initializer, data []core.Elemt) (impl Impl) {
	impl = Impl{
		buffer:      core.NewDataBuffer(data, conf.FrameSize),
		initializer: initializer,
		distrib:     distrib,
	}
	impl.uniform = distuv.Uniform{Max: 1, Min: 0, Src: conf.RGen}
	impl.store = NewCenterStore(impl.buffer, space, conf.RGen)
	impl.strategy = SeqStrategy{Buffer: impl.buffer}

	return
}

// Run executes the algorithm
func (impl *Impl) Run(conf core.Conf, space core.Space, closing <-chan bool) {
	var mcmcConf = conf.(Conf)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: impl.centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, impl.centroids),
		pdf:     impl.proba(impl.centroids, impl.centroids),
	}

	for i, loop := 0, true; i < mcmcConf.Iter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			current = impl.doIter(mcmcConf, space, current)
			impl.buffer.Apply()
		}
	}
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (impl *Impl) doIter(conf Conf, space core.Space, current proposal) proposal {
	var prop = impl.propose(conf, space, current)

	if impl.accept(conf, current, prop) {
		current = prop
		algo.Clust = prop.centers
		impl.store.SetCenters(algo.Clust)
		impl.acc++
	}

	impl.iter++
	return current
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal) (prop proposal) {
	var k = impl.nextK(current.k)
	var centers = impl.store.GetCenters(prop.k, impl.Clust)
	centers = impl.strategy.Iterate(centers, 1)
	centers = impl.alter(centers)
	prop = proposal{
		k:       k,
		centers: centers,
		loss:    impl.strategy.Loss(conf, space, centers),
		pdf:     impl.proba(centers, centers),
	}
	return
}

func (impl *Impl) accept(conf Conf, current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = conf.L2B() * float64(conf.Dim*(current.k-prop.k))
	var rGibbs = conf.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return impl.uniform.Rand() < rho
}

func (impl *Impl) nextK(conf Conf, k int) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(conf.ProbaK, conf.RGen)]

	switch {
	case newK < 1:
		return 1
	case newK > conf.MaxK:
		return conf.MaxK
	case newK > len(impl.data.Data):
		return len(impl.data.Data)
	default:
		return newK
	}
}

func (impl *Impl) alter(clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))

	for i := range clust {
		result[i] = impl.distrib.Sample(clust[i])
	}

	return result
}

func (impl *Impl) proba(x, mu core.Clust) (p float64) {
	p = 0.
	for i := range x {
		p += impl.distrib.Pdf(mu[i], x[i])
	}
	return p
}

// AcceptRatio returns ratio between acc and iter
func (impl *Impl) AcceptRatio() float64 {
	return float64(impl.acc) / float64(impl.iter)
}
