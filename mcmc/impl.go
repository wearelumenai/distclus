package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Impl of MCMC
type Impl struct {
	buffer      core.Buffer
	initializer core.Initializer
	strategy    Strategy
	uniform     distuv.Uniform
	distrib     Distrib
	store       CenterStore
	iter, acc   int
}

// Strategy specifies strategy methods
type Strategy interface {
	Iterate(Conf, core.Space, core.Clust, []core.Elemt, int) core.Clust
	Loss(Conf, core.Space, core.Clust, []core.Elemt) float64
}

// Init initializes the algorithm
func (impl *Impl) Init(conf core.ImplConf, space core.Space) (core.Clust, error) {
	var mcmcConf = conf.(Conf)
	impl.buffer.Apply()
	return impl.initializer(mcmcConf.InitK, impl.buffer.Data(), space, mcmcConf.RGen)
}

// NewImpl function
func NewImpl(conf *Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) (impl Impl) {
	SetConfigDefaults(conf)
	Verify(*conf)
	var buffer = core.NewDataBuffer(data, conf.FrameSize)
	impl = Impl{
		buffer:      buffer,
		initializer: initializer,
		distrib:     distrib,
		uniform:     distuv.Uniform{Max: 1, Min: 0, Src: conf.RGen},
		store:       NewCenterStore(conf.RGen),
		strategy:    &SeqStrategy{},
	}

	return
}

func (impl *Impl) initRun(conf *Conf, space core.Space, data []core.Elemt) {
	if impl.distrib == nil {
		if conf.Dim == 0 {
			conf.Dim = space.Dim(data)
		}
		impl.distrib = NewMultivT(MultivTConf{*conf})
	}
}

// Run executes the algorithm
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier func(core.Clust), closing <-chan bool) (err error) {
	var mcmcConf = conf.(Conf)
	var data = impl.buffer.Data()
	impl.initRun(&mcmcConf, space, data)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, centroids, data),
		pdf:     impl.proba(mcmcConf, space, centroids, centroids),
	}

	for i, loop := 0, true; i < mcmcConf.McmcIter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			data = impl.buffer.Data()
			current, centroids = impl.doIter(mcmcConf, space, current, centroids, data)
			notifier(centroids)
			err = impl.buffer.Apply()
		}
	}
	return
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() error {
	return impl.buffer.SetAsync()
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt) error {
	return impl.buffer.Push(elemt)
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (impl *Impl) doIter(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt) (proposal, core.Clust) {
	var prop = impl.propose(conf, space, current, centroids, data)

	if impl.accept(conf, current, prop) {
		current = prop
		centroids = prop.centers
		impl.store.SetCenters(centroids)
		impl.acc++
	}

	impl.iter++
	return current, centroids
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt) proposal {
	var k = impl.nextK(conf, current.k, data)
	var centers = impl.store.GetCenters(data, space, k, centroids)
	centers = impl.strategy.Iterate(conf, space, centers, data, 1)
	var alteredCenters = impl.alter(conf, space, centers)
	return proposal{
		k:       k,
		centers: alteredCenters,
		loss:    impl.strategy.Loss(conf, space, alteredCenters, data),
		pdf:     impl.proba(conf, space, alteredCenters, centers),
	}
}

func (impl *Impl) accept(conf Conf, current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = conf.L2B() * float64(conf.Dim*(current.k-prop.k))
	var rGibbs = conf.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return impl.uniform.Rand() < rho
}

func (impl *Impl) nextK(conf Conf, k int, data []core.Elemt) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(conf.ProbaK, conf.RGen)]

	switch {
	case newK < 1:
		return 1
	case newK > conf.MaxK:
		return conf.MaxK
	case newK > len(data):
		return len(data)
	default:
		return newK
	}
}

func (impl *Impl) alter(conf Conf, space core.Space, clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))

	for i, c := range clust {
		result[i] = impl.distrib.Sample(c)
	}

	return result
}

func (impl *Impl) proba(conf Conf, space core.Space, x, mu core.Clust) (p float64) {
	p = 0.
	for i, v := range x {
		p += impl.distrib.Pdf(mu[i], v)
	}
	return p
}

// AcceptRatio returns ratio between acc and iter
func (impl *Impl) AcceptRatio() float64 {
	return float64(impl.acc) / float64(impl.iter)
}
