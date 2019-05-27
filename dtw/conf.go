// Package dtw allows to computes DTW distance based clusters.
package dtw

import "distclus/core"

// Conf defines series configuration
type Conf struct {
	InnerSpace core.Space
	Window     int
}
