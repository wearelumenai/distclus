package kmeans

import (
	"distclus/core"
	"distclus/figures"
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

// Init Algorithm
func (impl *Impl) Init(conf core.ImplConf, space core.Space, _ core.Clust) (clust core.Clust, err error) {
	var kmeansConf = conf.(*Conf)
	_ = impl.buffer.Apply()
	return impl.initializer(kmeansConf.K, impl.buffer.Data(), space, kmeansConf.RGen)
}

// Iterate the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Iterate(conf core.ImplConf, space core.Space, centroids core.Clust) (clust core.Clust, runtimeFigures figures.RuntimeFigures, err error) {
	return impl.strategy.Iterate(space, centroids, impl.buffer.Data()),
		nil,
		impl.buffer.Apply()
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt, running bool) error {
	return impl.buffer.Push(elemt, running)
}
