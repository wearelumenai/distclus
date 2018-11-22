package kmeans

import (
	"distclus/core"
	"errors"
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
func NewImpl(conf Conf, initializer core.Initializer, data []core.Elemt) (impl Impl) {
	impl = Impl{
		buffer:      core.NewDataBuffer(data, conf.FrameSize),
		strategy:    &SeqStrategy{},
		initializer: initializer,
	}
	return
}

// Init Algorithm
func (impl *Impl) Init(conf core.Conf, space core.Space) (centroids core.Clust, err error) {
	var kmeansConf = conf.(Conf)
	SetConfigDefaults(&kmeansConf)
	Verify(kmeansConf)
	impl.buffer.Apply()
	var initialized bool
	centroids, initialized = impl.initializer(kmeansConf.K, impl.buffer.Data, space, kmeansConf.RGen)
	if !initialized {
		err = errors.New("Failed to initialize")
	} else {
		impl.centroids = centroids
	}
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
			impl.strategy.Iterate(space, impl.centroids, impl.buffer)
			impl.buffer.Apply()
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
func (impl *Impl) Push(elemt core.Elemt) (err error) {
	impl.buffer.Push(elemt)
	return
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync() (err error) {
	impl.buffer.SetAsync()
	return
}
