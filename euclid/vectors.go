package euclid

import (
	"distclus/core"
	"math"
)

// Space for vectors ([]float64)
type Space struct{}

// NewSpace creates a new Space
func NewSpace(conf Conf) Space {
	return Space{}
}

// Dist computes euclidean distance between two nodes
func (space Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)
	var sum = 0.
	for i := 0; i < len(e1); i++ {
		var v = e1[i] - e2[i]
		sum += v * v
	}
	return math.Sqrt(sum)
}

// Combine computes combination between two nodes
func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) core.Elemt {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)

	var dim = len(e1)
	var w1 = float64(weight1)
	var w2 = float64(weight2)
	var t = w1 + w2

	var result = make([]float64, dim)

	for i := 0; i < dim; i++ {
		result[i] = (e1[i]*w1 + e2[i]*w2) / t
	}

	return result
}

// Copy creates a copy of a vector
func (space Space) Copy(elemt core.Elemt) core.Elemt {
	var rv = elemt.([]float64)
	var crv = make([]float64, len(rv))
	copy(crv, rv)
	return crv
}

// Dim returns input data dimension
func (space Space) Dim(data []core.Elemt) (dim int) {
	if len(data) > 0 {
		elemts := data[0].([]float64)
		dim = len(elemts)
	}
	return
}
