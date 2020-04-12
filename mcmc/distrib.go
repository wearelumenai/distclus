package mcmc

import "go.lumenai.fr/distclus/v0/core"

// Distrib defines distribution methods
type Distrib interface {
	Sample(mu core.Elemt, time int) core.Elemt
	Pdf(x, mu core.Elemt, time int) float64
}
