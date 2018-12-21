package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"math"
	"time"

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
	if conf.Dim == 0 {
		conf.Dim = space.Dim(data)
	}
	if impl.distrib == nil {
		impl.distrib = NewMultivT(MultivTConf{*conf})
	}
}

// Run executes the algorithm
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier func(core.Clust), closing <-chan bool, closed chan<- bool) (err error) {
	var mcmcConf = conf.(Conf)
	var data = impl.buffer.Data()
	impl.initRun(&mcmcConf, space, data)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, centroids, data),
		pdf:     impl.proba(mcmcConf, space, centroids, centroids, impl.getCurrentTime(data)),
	}

	for i, loop := 0, true; i < mcmcConf.McmcIter && loop; i++ {
		select {
		case <-closing:
			loop = false
			closed <- true
			time.Sleep(300 * time.Millisecond)

		default:
			data = impl.buffer.Data()
			current, centroids = impl.doIter(mcmcConf, space, current, centroids, data, impl.getCurrentTime(data))
			notifier(centroids)
			err = impl.buffer.Apply()
		}
	}
	return
}

func (impl *Impl) getCurrentTime(data []core.Elemt) int {
	return len(data)
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

func (impl *Impl) doIter(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt, time int) (proposal, core.Clust) {
	var prop = impl.propose(conf, space, current, centroids, data, time)

	if impl.accept(conf, current, prop, time) {
		current = prop
		centroids = prop.centers
		impl.store.SetCenters(centroids)
		impl.acc++
	}

	impl.iter++
	return current, centroids
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt, time int) proposal {
	var k = impl.nextK(conf, current.k, data)
	var centers = impl.store.GetCenters(data, space, k, centroids)
	centers = impl.strategy.Iterate(conf, space, centers, data, 1)
	var alteredCenters = impl.alter(conf, space, centers, time)
	return proposal{
		k:       k,
		centers: alteredCenters,
		loss:    impl.strategy.Loss(conf, space, alteredCenters, data),
		pdf:     impl.proba(conf, space, alteredCenters, centers, time),
	}
}

func (impl *Impl) accept(conf Conf, current proposal, prop proposal, time int) bool {
	var rProp = current.pdf - prop.pdf
	var l2b = math.Log(2 * conf.B)
	var rInit = l2b * float64(conf.Dim*(current.k-prop.k))
	var lambda = conf.Amp * math.Sqrt(float64(conf.Dim+3)/float64(time))
	var rGibbs = lambda * (current.loss - prop.loss)

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

func (impl *Impl) alter(conf Conf, space core.Space, clust core.Clust, time int) core.Clust {
	var result = make(core.Clust, len(clust))

	for i, c := range clust {
		result[i] = impl.distrib.Sample(c, time)
	}

	return result
}

func (impl *Impl) proba(conf Conf, space core.Space, x, mu core.Clust, time int) (p float64) {
	p = 0.
	for i, v := range x {
		p += impl.distrib.Pdf(mu[i], v, time)
	}
	return p
}

// AcceptRatio returns ratio between acc and iter
func (impl *Impl) AcceptRatio() float64 {
	return float64(impl.acc) / float64(impl.iter)
}
