package mcmc

import (
	"distclus/core"
	"sync"
)

type DistribBuilder func(elemt core.Elemt) Distrib

type LazyDistrib struct {
	initializer DistribBuilder
	initialized bool
	mu          sync.Mutex
	distrib     Distrib
}

func NewLazyDistrib(initializer DistribBuilder) *LazyDistrib {
	return &LazyDistrib{
		initializer: initializer,
		mu:          sync.Mutex{},
	}
}

func (d *LazyDistrib) Sample(mu core.Elemt, time int) core.Elemt {
	d.initialize(mu)
	return d.distrib.Sample(mu, time)
}

func (d *LazyDistrib) Pdf(x, mu core.Elemt, time int) float64 {
	d.initialize(mu)
	return d.distrib.Pdf(x, mu, time)
}

func (d *LazyDistrib) initialize(elemt core.Elemt) {
	if !d.initialized {
		d.tryInitialize(elemt)
	}
}

func (d *LazyDistrib) tryInitialize(elemt core.Elemt) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.initialized {
		d.distrib = d.initializer(elemt)
		d.initialized = true
	}
}
