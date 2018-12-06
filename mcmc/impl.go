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
	Iterate(Conf, core.Space, core.Clust, core.Buffer, int) core.Clust
	Loss(Conf, core.Space, core.Clust, core.Buffer) float64
}

// Init initializes the algorithm
func (impl *Impl) Init(conf core.Conf, space core.Space) (core.Clust, error) {
	var mcmcConf = conf.(Conf)
	impl.buffer.Apply()
	return impl.initializer(mcmcConf.InitK, impl.buffer.Data(), space, mcmcConf.RGen)
}

// Reset this implementation
func (impl *Impl) Reset(conf *core.Conf, data []core.Elemt) (res core.Impl, err error) {
	var mcmcConf = (*conf).(Conf)
	if data == nil {
		data = impl.buffer.Data()
	}
	var _impl Impl
	if mcmcConf.Par {
		_impl = NewParImpl(&mcmcConf, impl.initializer, data, impl.distrib)
	} else {
		_impl = NewSeqImpl(&mcmcConf, impl.initializer, data, impl.distrib)
	}
	res = &_impl
	return
}

// NewImpl function
func NewImpl(conf *Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) (impl Impl) {
	SetConfigDefaults(conf)
	Verify(*conf)
	var buffer = core.NewDataBuffer(data, conf.FrameSize)
	if distrib == nil {
		distrib = NewMultivT(MultivTConf{*conf})
	}
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

// Run executes the algorithm
func (impl *Impl) Run(conf core.Conf, space core.Space, centroids core.Clust, notifier func(core.Clust), closing <-chan bool) (err error) {
	var mcmcConf = conf.(Conf)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, centroids, impl.buffer),
		pdf:     impl.proba(centroids, centroids),
	}

	for i, loop := 0, true; i < mcmcConf.McmcIter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			current, centroids = impl.doIter(mcmcConf, space, current, centroids)
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

func (impl *Impl) doIter(conf Conf, space core.Space, current proposal, centroids core.Clust) (proposal, core.Clust) {
	var prop = impl.propose(conf, space, current, centroids)

	if impl.accept(conf, current, prop) {
		current = prop
		centroids = prop.centers
		impl.store.SetCenters(centroids)
		impl.acc++
	}

	impl.iter++
	return current, centroids
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal, centroids core.Clust) proposal {
	var k = impl.nextK(conf, current.k)
	var centers = impl.store.GetCenters(impl.buffer, space, k, centroids)
	centers = impl.strategy.Iterate(conf, space, centers, impl.buffer, 1)
	var alteredCenters = impl.alter(centers)
	return proposal{
		k:       k,
		centers: alteredCenters,
		loss:    impl.strategy.Loss(conf, space, alteredCenters, impl.buffer),
		pdf:     impl.proba(alteredCenters, centers),
	}
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
	case newK > len(impl.buffer.Data()):
		return len(impl.buffer.Data())
	default:
		return newK
	}
}

func (impl *Impl) alter(clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))

	for i, c := range clust {
		result[i] = impl.distrib.Sample(c)
	}

	return result
}

func (impl *Impl) proba(x, mu core.Clust) (p float64) {
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
