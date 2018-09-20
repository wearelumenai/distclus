package mcmc

import "distclus/core"

type MCMCDistrib interface {
	Sample(mu core.Elemt) core.Elemt
	Pdf(x, mu core.Elemt) float64
}
