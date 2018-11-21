package mcmc

import "distclus/core"

// Distrib defines distribution methods
type Distrib interface {
	Sample(mu core.Elemt) core.Elemt
	Pdf(x, mu core.Elemt) float64
}
