package clustering_go

import (
	"math"
)

// Space for reals ([]float64)
type realSpace struct{}

// Check if a Elemt is contained in realSpace
func (space realSpace) check(elemt Elemt) []float64 {
	var n = elemt.([]float64)
	if len(n) == 0 {
		panic("Elemt is empty")
	}
	return n
}

// Check if two nodes are contained in the same realSpace(i.e. same dimension)
func (space realSpace) checkCombine(elemt1, elemt2 Elemt) ([]float64, []float64) {
	n1 := space.check(elemt1)
	n2 := space.check(elemt2)
	if len(n1) != len(n2) {
		panic("elemt1 and elemt2 have not the same length")
	}
	return n1, n2
}

// Compute euclidean distance between two nodes
func (space realSpace) dist(elemt1, elemt2 Elemt) float64 {
	e1, e2 := space.checkCombine(elemt1, elemt2)
	diff := make([]float64, len(e1))
	for i := 0; i < len(e1); i++ {
		diff[i] = e1[i] - e2[i]
	}
	var sum float64
	for _, val := range (diff) {
		sum += math.Pow(val, 2)
	}
	return math.Sqrt(sum)
}

// Compute combination between two nodes
func (space realSpace) combine(elemt1 Elemt, weight1 int, elemt2 Elemt, weight2 int) Elemt {
	e1, e2 := space.checkCombine(elemt1, elemt2)
	dim := len(e1)
	if weight1 == 0 && weight2 == 0 {
		panic("both weight are zero")
	}
	w1 := float64(weight1)
	w2 := float64(weight2)
	combination := make([]float64, dim)
	for i := 0; i < dim; i++ {
		combination[i] = (e1[i]*w1 + e2[i]*w2) / (w1 + w2)
	}
	return combination
}