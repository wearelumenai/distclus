package core

import (
	"golang.org/x/exp/rand"
	"errors"
	"math"
)

type ClustStatus int

const (
	Created ClustStatus = iota
	Running
	Closed
)

// Online Clustering algorithm.
// When a prediction is made, the element can be pushed to the model.
// A prediction consists in a centroid and a label.
// The following constraints must be met (otherwise an error is returned) :
// an element can't be pushed if the algorithm is closed,
// a prediction can't be done before the algorithm is run,
// no centroid can be returned before the algorithm is run.
type OnlineClust interface {
	Centroids() (Clust, error)
	Push(elemt Elemt) error
	Predict(elemt Elemt, push bool) (Elemt, int, error)
	Run(async bool)
	Close()
}

// Cluster centroids indexed by labels.
type Clust []Elemt

// Initializes k centroids from the given elements.
type Initializer = func(k int, elemts []Elemt, space Space, src *rand.Rand) (centroids Clust, success bool)

// Returns centroids and cardinalities in each clusters.
func (c Clust) AssignDBA(elemts []Elemt, space Space) (centroids Clust, cards []int) {
	centroids = make(Clust, len(c))
	cards = make([]int, len(c))

	for i, _ := range elemts {
		var _, ix, _ = c.Assign(elemts[i], space)

		if cards[ix] == 0 {
			centroids[ix] = space.Copy(elemts[i])
			cards[ix] = 1
		} else {
			space.Combine(centroids[ix], cards[ix], elemts[i], 1)
			cards[ix] += 1
		}
	}

	return
}

// Assigns elemts to each centroids
func (c Clust) AssignAll(elemts []Elemt, space Space) (clusters [][]Elemt) {
	clusters = make([][]Elemt, len(c))

	for _, elemt := range elemts {
		var idx, _ = c.nearest(elemt, space)
		clusters[idx] = append(clusters[idx], elemt)
	}

	return
}

// Returns the element nearest centroid, its label and the distance to the centroid
func (c Clust) Assign(elemt Elemt, space Space) (centroid Elemt, label int, dist float64) {
	label, dist = c.nearest(elemt, space)
	centroid = c[label]
	return
}

// Compute loss from distances between elements and their nearest centroid
func (c Clust) Loss(elemts []Elemt, space Space, norm float64) float64 {
	var sum = 0.

	for i, _ := range elemts {
		var _, min = c.nearest(elemts[i], space)
		sum += math.Pow(min, norm)
	}

	return sum
}

// Returns the label of element nearest centroid and the distance
func (c Clust) nearest(elemt Elemt, space Space) (label int, min float64) {
	min = space.Dist(elemt, c[0])
	label = 0

	for i := 1; i < len(c); i++ {
		var d = space.Dist(elemt, c[i])
		if min > d {
			min = d
			label = i
		}
	}

	return label, min
}

// Returns the averaged element
func DBA(elemts []Elemt, space Space) (dba Elemt, err error) {

	if len(elemts) == 0 {
		err = errors.New("DBA needs at least one elements")
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

// An initializer that always returns the centroids
func (c Clust) Initializer(int, []Elemt, Space, *rand.Rand) (Clust, bool) {
	return c, true
}
