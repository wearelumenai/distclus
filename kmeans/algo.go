package kmeans

import (
	"distclus/core"
)

// Impl algorithm abstract implementation
type Impl struct {
	strategy    Strategy
	buffer      core.DataBuffer
	centroids   core.Clust
	initializer core.Initializer
}

// Strategy Abstract Impl strategy to be implemented by concrete algorithms
type Strategy interface {
	Iterate(space core.Space, centroids core.Clust, buffer core.DataBuffer) core.Clust
}

// NewImpl returns a kmeans implementation
func NewImpl(conf Conf, data []core.Elemt, initializer core.Initializer) Impl {
	return Impl{
		buffer:      *core.NewDataBuffer(data, conf.FrameSize),
		strategy:    &SeqStrategy{},
		initializer: initializer,
	}
}

// Init Algorithm
func (impl *Impl) Init(conf core.Conf) (core.Clust, bool) {
	var kmeansConf = conf.(Conf)
	SetConfigDefaults(&kmeansConf)
	Verify(kmeansConf)
	impl.buffer.Apply()
	return impl.initializer(kmeansConf.K, impl.buffer.Data, kmeansConf.space, kmeansConf.RGen)
}

// Run the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Run(conf core.Conf, space core.Space, closing <-chan bool) {
	var kmeansConf = conf.(Conf)
	for iter, loop := 0, true; iter < kmeansConf.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false

		default:
			impl.strategy.Iterate(space, impl.centroids, impl.buffer)
			impl.buffer.Apply()
		}
	}
}

// Centroids returns a copy of impl centroids
func (impl *Impl) Centroids() (centroids core.Clust) {
	centroids = make(core.Clust, len(impl.centroids))
	copy(centroids, impl.centroids)
	return
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt) {
	impl.buffer.Push(elemt)
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() {
	impl.buffer.SetAsync()
}
