package kmeans

import (
	"distclus/core"
)

// Impl algorithm abstract implementation
type Impl struct {
	strategy    Strategy
	buffer      core.Buffer
	centroids   core.Clust
	initializer core.Initializer
}

// Strategy Abstract Impl strategy to be implemented by concrete algorithms
type Strategy interface {
	Iterate(space core.Space, centroids core.Clust, buffer core.Buffer) core.Clust
}

// NewImpl returns a kmeans implementation
func NewImpl(conf Conf, initializer core.Initializer, data []core.Elemt) (impl Impl) {
	SetConfigDefaults(&conf)
	Verify(conf)
	return Impl{
		buffer:      core.NewDataBuffer(data, conf.FrameSize),
		strategy:    &SeqStrategy{},
		initializer: initializer,
	}
}

// Init Algorithm
func (impl *Impl) Init(conf core.Conf, space core.Space) (err error) {
	var kmeansConf = conf.(Conf)
	impl.buffer.Apply()
	impl.centroids, err = impl.initializer(kmeansConf.K, impl.buffer.Data(), space, kmeansConf.RGen)
	return
}

// Run the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Run(conf core.Conf, space core.Space, closing <-chan bool) (err error) {
	var kmeansConf = conf.(Conf)
	for iter, loop := 0, true; iter < kmeansConf.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false

		default:
			impl.centroids = impl.strategy.Iterate(space, impl.centroids, impl.buffer)
			err = impl.buffer.Apply()
		}
	}
	return
}

// Centroids returns a copy of impl centroids
func (impl *Impl) Centroids() (centroids core.Clust, err error) {
	centroids = make(core.Clust, len(impl.centroids))
	copy(centroids, impl.centroids)
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
