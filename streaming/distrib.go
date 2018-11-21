package streaming

import "distclus/core"

type StreamingDistrib interface {
	Sample(mu core.Elemt) core.Elemt
	Pdf(x, mu core.Elemt) float64
}
