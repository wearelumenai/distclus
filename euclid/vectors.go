// Package euclid allows to computes Euclidean distance based clusters.
package euclid

import (
	"math"

	"github.com/wearelumenai/distclus/core"
)

// Space for vectors ([]float64)
type Space struct{}

// NewSpace creates a new Space
func NewSpace() Space {
	return Space{}
}

// Dist computes euclidean distance between two nodes
func (space Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)
	return space.PointDist(e1, e2)
}

// PointDist returns distance points
func (space Space) PointDist(point1 []float64, point2 []float64) float64 {
	var sum = 0.
	for i := 0; i < len(point1); i++ {
		var v = point1[i] - point2[i]
		sum += v * v
	}
	return math.Sqrt(sum)
}

// Combine computes combination between two nodes
func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) core.Elemt {
	var e1 = elemt1.([]float64)
	var e2 = elemt2.([]float64)

	return space.PointCombine(e1, weight1, e2, weight2)
}

// PointCombine returns combination of points
func (space Space) PointCombine(point1 []float64, weight1 int, point2 []float64, weight2 int) []float64 {
	var dim = len(point1)
	var w1 = float64(weight1)
	var w2 = float64(weight2)
	var t = w1 + w2
	var result = make([]float64, dim)
	for i := 0; i < dim; i++ {
		result[i] = (point1[i]*w1 + point2[i]*w2) / t
	}
	return result
}

// Copy creates a copy of a vector
func (space Space) Copy(elemt core.Elemt) core.Elemt {
	var point = elemt.([]float64)
	return space.PointCopy(point)
}

// PointCopy copy points
func (space Space) PointCopy(point []float64) []float64 {
	var newPoint = make([]float64, len(point))
	copy(newPoint, point)
	return newPoint
}

// Dim returns input data dimension
func (space Space) Dim(data []core.Elemt) (dim int) {
	if len(data) > 0 {
		elemts := data[0].([]float64)
		dim = len(elemts)
	}
	return
}
