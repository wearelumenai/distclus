package algo

import (
	"golang.org/x/exp/rand"
	"distclus/core"
	"errors"
	"math"
)

type ClustStatus int

const (
	Created ClustStatus = iota
	Running
	Closed
)

// Online Clust algorithm interface.
type OnlineClust interface {
	// Add an element to Clust Data set.
	Push(elemt core.Elemt) error
	// Return model current centroids configuration.
	Centroids() (Clust, error)
	// Make a prediction on a element and return the associated center and its index.
	Predict(elemt core.Elemt, push bool) (core.Elemt, int, error)
	// Run Clust algorithm.
	Run(async bool)
	// Close algorithm Clust process.
	Close()
}

// Indexed clustering result
type Clust []core.Elemt

// AssignAll testElemts on elemts at each centers
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
		var min = space.Dist(elemt, c[0])
		for i := 1; i < len(c); i++ {
			var d = space.Dist(elemt, c[i])
			if min > d {
				min = d
			}
		}

		sum += math.Pow(min, norm)
	}
	return sum / float64(len(data))
}

// Returns the index of the closest element to elemt in elemts.
func assign(elemt core.Elemt, clust Clust, space core.Space) (int, float64) {

	if len(clust) < 1 {
		panic("empty clust")
	}

	distances := make([]float64, len(clust))
	for i, node := range clust {
		distances[i] = space.Dist(elemt, node)
	}

	var lowest = distances[0]
	var index int
	for i := 1; i < len(distances); i++ {
		if distances[i] < lowest {
			lowest = distances[i]
			index = i
		}
	}

	return index, lowest
}

// Return the DBA of nodes based on the core.Space combination method.
// If nodes are empty function panic.
func DBA(elemts []core.Elemt, space core.Space) (dba core.Elemt, err error) {

	if l := len(elemts); l < 1 {
		err = errors.New("elemts are empty")
		return
	}

	dba = space.Copy(elemts[0])
	var weight = 1

	for i := 1; i < len(elemts); i++ {
		space.Combine(dba, weight, elemts[i], 1)
		weight += 1
	}

	return
}

func (c Clust) Initializer(int, []core.Elemt, core.Space, *rand.Rand) Clust {
	return c
}
