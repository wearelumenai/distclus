package core

import (
	"math"
	"golang.org/x/exp/rand"
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
	// Add an element to Clust data set.
	Push(elemt Elemt)
	// Return model current centroids configuration.
	Centroids() (Clust, error)
	// Make a prediction on a element and return the associated center and its index.
	Predict(elemt Elemt, push bool) (Elemt, int, error)
	// Run Clust algorithm.
	Run()
	// Close algorithm Clust process.
	Close()
}

// Indexed clustering result
type Clust []Elemt

// Assign elements on elemts at each centers
func (c Clust) Assign(elemts []Elemt, space Space) [][]Elemt {
	var clusters = make([][]Elemt, len(c))
	for _, elemt := range elemts {
		var idx = assign(elemt, c, space)
		clusters[idx] = append(clusters[idx], elemt)
	}
	return clusters
}

// Assign a element to a center and return the center and its index
func (c Clust) UAssign(elemt Elemt, space Space) (center Elemt, idx int) {
	idx = assign(elemt, c, space)
	return c[idx], idx
}

// Compute loss of centers configuration with given data
func (c Clust) Loss(data []Elemt, space Space, norm float64) float64 {
	var sum float64
	for _, elemt := range data {
		var min = math.MaxFloat64
		for _, center := range c {
			min = math.Min(min, math.Pow(space.dist(elemt, center), norm))
		}
		sum += min
	}
	return sum / float64(len(data))
}

// Returns the index of the closest element to elemt in elemts.
func assign(elemt Elemt, elemts []Elemt, space Space) int {
	if len(elemts) < 1 {
		panic("elemts collection is empty")
	}
	distances := make([]float64, len(elemts))
	for i, node := range elemts {
		distances[i] = space.dist(elemt, node)
	}
	current := distances[0]
	var index int
	for i, dist := range distances {
		if dist < current {
			current = dist
			index = i
		}
	}
	return index
}

// Return the mean of nodes based on the Space combination method.
// If nodes are empty function panic.
func mean(elemts []Elemt, space Space) Elemt {
	l := len(elemts)
	if l < 1 {
		panic("elemts are empty")
	}
	mean := elemts[0]
	weight := 1
	for _, node := range elemts {
		mean = space.combine(node, 1, mean, weight)
		weight += 1
	}
	return mean
}

func (c Clust) initializer(k int, nodes []Elemt, space Space, src *rand.Rand) Clust {
	return c
}