package algo

import (
	"math"
	"golang.org/x/exp/rand"
	"distclus/core"
)

type ClustStatus int

const (
	Created     ClustStatus = iota
	Initialized
	Running
	Closed
)

// Online Clust algorithm interface.
type OnlineClust interface {
	// Add an element to Clust Data set.
	Push(elemt core.Elemt)
	// Return model current centroids configuration.
	Centroids() (Clust, error)
	// Make a prediction on a element and return the associated center and its index.
	Predict(elemt core.Elemt, push bool) (core.Elemt, int, error)
	// Run Clust algorithm.
	Run()
	// Close algorithm Clust process.
	Close()
}

// Indexed clustering result
type Clust []core.Elemt

// AssignAll elements on elemts at each centers
func (c Clust) AssignAll(elemts []core.Elemt, space core.Space) [][]core.Elemt {
	var clusters = make([][]core.Elemt, len(c))
	for _, elemt := range elemts {
		var idx, _ = assign(elemt, c, space)
		clusters[idx] = append(clusters[idx], elemt)
	}
	return clusters
}

// AssignAll a element to a center and return the center and its index
func (c Clust) Assign(elemt core.Elemt, space core.Space) (core.Elemt, int, float64) {
	var idx, dist = assign(elemt, c, space)
	return c[idx], idx, dist
}

// Compute loss of centers configuration with given Data
func (c Clust) Loss(data []core.Elemt, space core.Space, norm float64) float64 {
	var sum float64
	for _, elemt := range data {
		var min = math.MaxFloat64
		for _, center := range c {
			min = math.Min(min, math.Pow(space.Dist(elemt, center), norm))
		}
		sum += min
	}
	return sum / float64(len(data))
}

// Returns the index of the closest element to elemt in elemts.
func assign(elemt core.Elemt, elemts []core.Elemt, space core.Space) (int, float64) {
	if len(elemts) < 1 {
		panic("elemts collection is empty")
	}

	distances := make([]float64, len(elemts))
	for i, node := range elemts {
		distances[i] = space.Dist(elemt, node)
	}

	var lowest = distances[0]
	var index int
	for i, dist := range distances {
		if dist < lowest {
			lowest = dist
			index = i
		}
	}

	return index, lowest
}

// Return the DBA of nodes based on the core.Space combination method.
// If nodes are empty function panic.
func DBA(elemts []core.Elemt, space core.Space) core.Elemt {

	if l := len(elemts); l < 1 {
		panic("elemts are empty")
	}

	var mean = elemts[0]
	var weight = 1

	for i:=1; i<len(elemts); i++ {
		mean = space.Combine(elemts[i], 1, mean, weight)
		weight += 1
	}

	return mean
}

func (c Clust) Initializer(k int, nodes []core.Elemt, space core.Space, src *rand.Rand) Clust {
	return c
}