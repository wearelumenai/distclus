package mcmc

import "distclus/core"

// Distrib defines distribution methods
type Distrib interface {
	Sample(mu core.Elemt, time int) core.Elemt
	Pdf(x, mu core.Elemt, time int) float64
}
