package mcmc

import (
	"sync"

	"github.com/wearelumenai/distclus/v0/core"
)

// DistribBuilder represents functor that build a Distrib from data
type DistribBuilder func(elemt core.Elemt) Distrib

// LateDistrib initializes a wrapped Distrib when the first data is known
type LateDistrib struct {
	initializer DistribBuilder
	initialized bool
	mu          sync.Mutex
	distrib     Distrib
}

// NewLateDistrib creates a new LateDistrib instance
func NewLateDistrib(initializer DistribBuilder) *LateDistrib {
	return &LateDistrib{
		initializer: initializer,
		mu:          sync.Mutex{},
	}
}

// Sample initializes the wrapped Distrib if necessary then forward to its Sample method
func (d *LateDistrib) Sample(mu core.Elemt, time int) core.Elemt {
	d.initialize(mu)
	return d.distrib.Sample(mu, time)
}

// Pdf initializes the wrapped Distrib if necessary then forward to its Pdf method
func (d *LateDistrib) Pdf(x, mu core.Elemt, time int) float64 {
	d.initialize(mu)
	return d.distrib.Pdf(x, mu, time)
}

func (d *LateDistrib) initialize(elemt core.Elemt) {
	if !d.initialized {
		d.tryInitialize(elemt)
	}
}

func (d *LateDistrib) tryInitialize(elemt core.Elemt) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.initialized {
		d.distrib = d.initializer(elemt)
		d.initialized = true
	}
}
