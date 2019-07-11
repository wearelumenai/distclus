package mcmc

import (
	"distclus/core"
	"distclus/figures"
	"distclus/kmeans"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"time"
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
	forever     bool
	dim         int
}

// Strategy specifies strategy methods
type Strategy interface {
	Iterate(Conf, core.Space, core.Clust, []core.Elemt, int) core.Clust
	Loss(Conf, core.Space, core.Clust, []core.Elemt) float64
}

// Init initializes the algorithm
func (impl *Impl) Init(conf core.ImplConf, space core.Space) (core.Clust, error) {
	var mcmcConf = conf.(Conf)
	_ = impl.buffer.Apply()
	impl.iter = 0
	return impl.initializer(mcmcConf.InitK, impl.buffer.Data(), space, mcmcConf.RGen)
}

// Run executes the algorithm
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier core.Notifier, closing <-chan bool, closed chan<- bool) (err error) {
	var mcmcConf = conf.(Conf)
	var data = impl.buffer.Data()
	impl.dim = space.Dim(centroids)
	var current = proposal{
		k:       mcmcConf.InitK,
		centers: centroids,
		loss:    impl.strategy.Loss(mcmcConf, space, centroids, data),
		pdf:     impl.proba(mcmcConf, space, centroids, centroids, impl.getCurrentTime(data)),
	}

	var start = time.Now()
	for loop := impl.forever || impl.iter < mcmcConf.McmcIter; loop; {
		select {
		case <-closing:
			closed <- true
			time.Sleep(300 * time.Millisecond)
			loop = false

		default:
			impl.iter++
			data = impl.buffer.Data()
			current, centroids = impl.doIter(mcmcConf, space, current, centroids, data, impl.getCurrentTime(data))
			notifier(centroids, impl.runtimeFigures())
			err = impl.buffer.Apply()
			if time.Now().Sub(start).Seconds() > float64(mcmcConf.Timeout) {
				err = core.ErrTimeOut
				loop = false
			} else {
				loop = impl.forever || impl.iter < mcmcConf.McmcIter
			}
		}
	}
	return
}

func (impl *Impl) getCurrentTime(data []core.Elemt) int {
	return len(data)
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() error {
	impl.forever = true
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

	return current, centroids
}

func (impl *Impl) propose(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt, time int) proposal {
	k, centers := impl.getKCenters(conf, space, current, centroids, data)
	centers = impl.alter(conf, space, centers, time)
	centers = impl.strategy.Iterate(conf, space, centers, data, 1)
	return proposal{
		k:       k,
		centers: centers,
		loss:    impl.strategy.Loss(conf, space, centers, data),
		pdf:     impl.proba(conf, space, centers, centers, time),
	}
}

func (impl *Impl) getKCenters(conf Conf, space core.Space, current proposal, centroids core.Clust, data []core.Elemt) (int, core.Clust) {
	var k = impl.nextK(conf, current.k, data)
	var centers, err = impl.store.GetCenters(data, space, k, centroids)
	if err != nil {
		k = current.k
		centers, _ = impl.store.GetCenters(data, space, k, centroids)
	}
	return k, centers
}

func (impl *Impl) accept(conf Conf, current proposal, prop proposal, time int) bool {
	var rProp = current.pdf - prop.pdf
	var l2b = math.Log(2 * conf.B)
	var rInit = l2b * float64(impl.dim*(current.k-prop.k))
	var lambda = conf.Amp * math.Sqrt(float64(impl.dim+3)/float64(time))
	var rGibbs = lambda * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return impl.uniform.Rand() < rho
}

func (impl *Impl) nextK(conf Conf, k int, data []core.Elemt) int {
	var i, _ = kmeans.WeightedChoice(conf.ProbaK, conf.RGen)
	var newK = k + []int{-1, 0, 1}[i]

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

// runtimeFigures returns specific kmeans properties
func (impl Impl) runtimeFigures() map[string]float64 {
	return map[string]float64{figures.Iterations: float64(impl.iter), figures.Acceptations: float64(impl.acc)}
}
