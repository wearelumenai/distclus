package core

import (
	"golang.org/x/exp/rand"
	"errors"
	"math"
)

type ClustStatus int

type Initializer = func(k int, elemts []Elemt, space Space, src *rand.Rand) (Clust, bool)

const (
	Created ClustStatus = iota
	Running
	Closed
)

// Online Clust algorithm interface.
type OnlineClust interface {
	// Add an element to Clust Data set.
	Push(elemt Elemt) error
	// Return model current centroids configuration.
	Centroids() (Clust, error)
	// Make a prediction on a element and return the associated center and its index.
	Predict(elemt Elemt, push bool) (Elemt, int, error)
	// Run Clust algorithm.
	Run(async bool)
	// Close algorithm Clust process.
	Close()
}

// Indexed clustering result
type Clust []Elemt

// AssignCount count elemts in each centers
func (c Clust) AssignDBA(elemts []Elemt, space Space) (Clust, []int) {
	var result = make(Clust, len(c))
	var cards = make([]int, len(c))

	for i, _ := range elemts {
		var _, ix, _ = c.Assign(elemts[i], space)

		if cards[ix] == 0 {
			result[ix] = space.Copy(elemts[i])
			cards[ix] = 1
		} else {
			space.Combine(result[ix], cards[ix], elemts[i], 1)
			cards[ix] += 1
		}
	}

	return result, cards
}

// AssignAll assign elemts to each centers
func (c Clust) AssignAll(elemts []Elemt, space Space) [][]Elemt {
	var clusters = make([][]Elemt, len(c))
	for _, elemt := range elemts {
		var idx, _ = assign(elemt, c, space)
		clusters[idx] = append(clusters[idx], elemt)
	}
	return clusters
}

// AssignAll a element to a center and return the center and its index
func (c Clust) Assign(elemt Elemt, space Space) (Elemt, int, float64) {
	var idx, dist = assign(elemt, c, space)
	return c[idx], idx, dist
}

// Compute loss of centers configuration with given Data
func (c Clust) Loss(data []Elemt, space Space, norm float64) float64 {
	var sum = 0.
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
	return sum
}

// Returns the index of the closest element to elemt in elemts.
func assign(elemt Elemt, clust Clust, space Space) (int, float64) {

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

// Return the DBA of nodes based on the Space combination method.
// If nodes are empty function panic.
func DBA(elemts []Elemt, space Space) (dba Elemt, err error) {

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

func (c Clust) Initializer(int, []Elemt, Space, *rand.Rand) (Clust, bool) {
	return c, true
}
