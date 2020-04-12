package mcmc

import (
	"go.lumenai.fr/distclus/v0/core"
)

// Dirac represents a Distrib that always returns the conditional.
type Dirac struct {
}

// NewDirac creates a new Dirac instance.
func NewDirac() Dirac {
	return Dirac{}
}

// Sample always returns mu
func (Dirac) Sample(mu core.Elemt, time int) core.Elemt {
	return mu
}

// Pdf returns 1 if mu == x
func (Dirac) Pdf(x, mu core.Elemt, time int) float64 {
	// this works because x == mu in all cases
	return 1
}
