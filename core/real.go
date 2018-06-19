package core

import (
	"math"
)

// Space for reals ([]float64)
type RealSpace struct{}

// Compute euclidean distance between two nodes
func (space RealSpace) Dist(elemt1, elemt2 Elemt) float64 {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)
	var sum = 0.
	for i := 0; i < len(e1); i++ {
		var v = e1[i] - e2[i]
		sum += v*v
	}
	return math.Sqrt(sum)
}

// Compute combination between two nodes
func (space RealSpace) Combine(elemt1 Elemt, weight1 int, elemt2 Elemt, weight2 int) {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)
	dim := len(e1)
	w1 := float64(weight1)
	w2 := float64(weight2)
	t := w1 + w2
	for i := 0; i < dim; i++ {
		e1[i] = (e1[i]*w1 + e2[i]*w2) / t
	}
}

// Create a copy of a vector
func (space RealSpace) Copy(elemt Elemt) Elemt {
	var rv = elemt.([]float64)
	var crv = make([]float64, len(rv))
	copy(crv, rv)
	return crv
}
