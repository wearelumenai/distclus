package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"errors"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Impl of MCMC
type Impl struct {
	centroids   core.Clust
	buffer      core.DataBuffer
	strategy    Strategy
	uniform     distuv.Uniform
	distrib     Distrib
	store       CenterStore
	iter, acc   int
	initializer core.Initializer
}

// Strategy specifies strategy methods
type Strategy interface {
	Iterate(Conf, core.Space, core.Clust, core.DataBuffer, int) core.Clust
	Loss(Conf, core.Space, core.Clust, core.DataBuffer) float64
}

// Init initializes the algorithm
func (impl *Impl) Init(conf core.Conf, space core.Space) (centroids core.Clust, err error) {
	var mcmcConf = conf.(Conf)
	SetConfigDefaults(&mcmcConf)
	Verify(mcmcConf)
	impl.buffer.Apply()
	var initialized bool
	centroids, initialized = impl.initializer(mcmcConf.InitK, impl.buffer.Data, space, mcmcConf.RGen)
	if !initialized {
		err = errors.New("Failed to initialize")
	} else {
		impl.centroids = centroids
	}
	return
}

// NewImpl function
func NewImpl(conf Conf, distrib Distrib, initializer core.Initializer, data []core.Elemt) (impl Impl) {
	var buffer = core.NewDataBuffer(data, conf.FrameSize)
	if distrib == nil {
		distrib = NewMultivT(
			MultivTConf{
				Conf: Conf{
					Dim: 1,
				},
			},
		)
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

// Centroids returns a copy of impl centroids
func (impl *Impl) Centroids() (centroids core.Clust, err error) {
	centroids = make(core.Clust, len(impl.centroids))
	copy(centroids, impl.centroids)
	return
}

// Run executes the algorithm
func (impl *Impl) Run(conf core.Conf, space core.Space, closing <-chan bool) (err error) {
	var mcmcConf = conf.(Conf)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: impl.centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, impl.centroids, impl.buffer),
		pdf:     impl.proba(impl.centroids, impl.centroids),
	}

	for i, loop := 0, true; i < mcmcConf.Iter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			current = impl.doIter(mcmcConf, space, current)
			err = impl.buffer.Apply()
		}
	}
	return
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() (err error) {
	impl.buffer.SetAsync()
	return
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt) (err error) {
	impl.buffer.Push(elemt)
	return
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
		impl.centroids = prop.centers
		impl.store.SetCenters(impl.centroids)
		impl.acc++
	}

	impl.iter++
	return current
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal) (prop proposal) {
	var k = impl.nextK(conf, current.k)
	var centers = impl.store.GetCenters(impl.buffer, space, prop.k, impl.centroids)
	centers = impl.strategy.Iterate(conf, space, centers, impl.buffer, 1)
	centers = impl.alter(centers)
	prop = proposal{
		k:       k,
		centers: centers,
		loss:    impl.strategy.Loss(conf, space, centers, impl.buffer),
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
	case newK > len(impl.buffer.Data):
		return len(impl.buffer.Data)
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
