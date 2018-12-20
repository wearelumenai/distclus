package kmeans

import (
	"distclus/core"
	"time"
)

// Impl algorithm abstract implementation
type Impl struct {
	strategy    Strategy
	buffer      core.Buffer
	initializer core.Initializer
}

// Strategy Abstract Impl strategy to be implemented by concrete algorithms
type Strategy interface {
	Iterate(space core.Space, centroids core.Clust, data []core.Elemt) core.Clust
}

// NewImpl returns a kmeans implementation
func NewImpl(conf *Conf, initializer core.Initializer, data []core.Elemt) (impl Impl) {
	SetConfigDefaults(conf)
	Verify(*conf)
	return Impl{
		buffer:      core.NewDataBuffer(data, conf.FrameSize),
		strategy:    &SeqStrategy{},
		initializer: initializer,
	}
}

// Init Algorithm
func (impl *Impl) Init(conf core.ImplConf, space core.Space) (core.Clust, error) {
	var kmeansConf = conf.(Conf)
	impl.buffer.Apply()
	return impl.initializer(kmeansConf.K, impl.buffer.Data(), space, kmeansConf.RGen)
}

// Run the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier func(core.Clust), closing <-chan bool, closed chan<- bool) (err error) {
	var kmeansConf = conf.(Conf)
	for iter, loop := 0, true; iter < kmeansConf.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false
			closed <- true
			time.Sleep(300 * time.Millisecond)

		default:
			centroids = impl.strategy.Iterate(space, centroids, impl.buffer.Data())
			notifier(centroids)
			err = impl.buffer.Apply()
		}
	}
	return
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt) error {
	return impl.buffer.Push(elemt)
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() error {
	return impl.buffer.SetAsync()
}
