package complex

import (
	"distclus/core"
	"math/cmplx"
)

// Space for complexes ([]complex128)
type Space struct{}

// NewSpace creates a new space
func NewSpace(conf core.Conf) Space {
	return Space{}
}

// Dist computes euclidean distance between two nodes
func (space Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var e1 = elemt1.([]complex128)
	var e2 = elemt2.([]complex128)
	var sum complex128
	for i := 0; i < len(e1); i++ {
		var v = e1[i] - e2[i]
		sum += v * v
	}
	return cmplx.Abs(cmplx.Sqrt(sum))
}

// Combine computes combination between two nodes
func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) {
	var e1 = elemt1.([]complex128)
	var e2 = elemt2.([]complex128)
	dim := len(e1)
	w1 := complex(float64(weight1), 0)
	w2 := complex(float64(weight2), 0)
	t := w1 + w2
	for i := 0; i < dim; i++ {
		e1[i] = (e1[i]*w1 + e2[i]*w2) / t
	}
}

// Copy creates a copy of a vector
func (space Space) Copy(elemt core.Elemt) core.Elemt {
	var rv = elemt.([]complex128)
	var crv = make([]complex128, len(rv))
	copy(crv, rv)
	return crv
}
