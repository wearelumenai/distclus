package clustering_go

import (
	"math"
)

// Node interface representing a clusterisable entity.
type node interface{}

// Space in which are defined nodes.
type space interface {
	dist(node1, node2 node) float64
	combine(node1 node, weight1 int, node2 node, weight2 int) node
}

// Space for reals ([]float64)
type realSpace struct{}

// Check if a node is contained in realSpace
func (space realSpace) check(node node) []float64 {
	n := node.([]float64)
	if len(n) == 0 {
		panic("node is empty")
	}
	return n
}

// Check if two nodes are contained in the same realSpace(i.e. same dimension)
func (space realSpace) checkCombine(node1, node2 node) ([]float64, []float64) {
	n1 := space.check(node1)
	n2 := space.check(node2)
	if len(n1) != len(n2) {
		panic("node1 and node2 have not the same length")
	}
	return n1, n2
}

// Compute euclidean distance between two nodes
func (space realSpace) dist(node1, node2 node) float64 {
	n1, n2 := space.checkCombine(node1, node2)
	diff := make([]float64, len(n1))
	for i := 0; i < len(n1); i++ {
		diff[i] = n1[i] - n2[i]
	}
	var sum float64
	for _, val := range (diff) {
		sum += math.Pow(val, 2)
	}
	return math.Sqrt(sum)
}

// Compute combination between two nodes
func (space realSpace) combine(node1 node, weight1 int, node2 node, weight2 int) node {
	n1, n2 := space.checkCombine(node1, node2)
	dim := len(n1)
	if weight1 == 0 && weight2 == 0 {
		panic("both weight are zero")
	}
	w1 := float64(weight1)
	w2 := float64(weight2)
	combination := make([]float64, dim)
	for i := 0; i < dim; i++ {
		combination[i] = (n1[i]*w1 + n2[i]*w2) / (w1 + w2)
	}
	return combination
}