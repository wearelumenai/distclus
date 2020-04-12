package template // to rename for specific algorithm

import (
	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/figures"
)

// Impl represents the algorithm implementation
type Impl struct {
}

// NewImpl creates a new Impl instance
func NewImpl(conf Conf, elemts []core.Elemt) Impl {
	return Impl{}
}

// Init initializes the algorithm
func (impl *Impl) Init(core.ImplConf, core.Space, core.Clust) (clust core.Clust, err error) {
	return
}

// Iterate runs one iteration of the algorithm
// If clust is nil, algorithm execution does not increment iterations
func (impl *Impl) Iterate(conf core.ImplConf, space core.Space, centroids core.Clust) (clust core.Clust, runtimeFigures figures.RuntimeFigures, err error) {
	return
}

// Push pushes a new element
func (impl *Impl) Push(elemt core.Elemt, running bool) (err error) {
	return
}

// Copy the impl
func (impl *Impl) Copy(core.ImplConf, core.Space) (core.Impl, error) {
	return impl, nil
}
