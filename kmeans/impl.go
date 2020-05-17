package kmeans

import (
	"github.com/wearelumenai/distclus/core"
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
func (impl *Impl) Init(model core.OCModel) (clust core.Clust, err error) {
	var kmeansConf = model.Conf().(*Conf)
	_ = impl.buffer.Apply()
	return impl.initializer(kmeansConf.K, impl.buffer.Data(), model.Space(), kmeansConf.RGen)
}

// Iterate the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Iterate(model core.OCModel) (clust core.Clust, runtimeFigures core.RuntimeFigures, err error) {
	return impl.strategy.Iterate(model.Space(), model.Centroids(), impl.buffer.Data()),
		nil,
		impl.buffer.Apply()
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt, model core.OCModel) error {
	return impl.buffer.Push(elemt, model.Status().Alive())
}

// Copy impl
func (impl *Impl) Copy(model core.OCModel) (core.Impl, error) {
	var newConf = model.Conf().(*Conf)
	var algo = NewAlgo(*newConf, model.Space(), impl.buffer.Data(), impl.initializer)
	return algo.Impl(), nil
}
