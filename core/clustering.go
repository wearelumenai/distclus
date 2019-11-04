package core

import (
	"errors"
	"math"

	"github.com/gonum/floats"

	"golang.org/x/exp/rand"
)

// ClustStatus integer type
type ClustStatus = int64

// ClustStatus const values
const (
	Created  ClustStatus = iota
	Ready                //
	Running              // used when algorithm run or after a pushed data when algo is in paused status
	Idle                 // paused by user
	Sleeping             // only in online clustering mode when no data and iterations are done
	Failed               // if an error occured during execution
	Closed               // only in online clustering mode
)

// Clust type is an abbrevation for centroids indexed by labels.
type Clust []Elemt

// Initializer function initializes k centroids from the given elements.
type Initializer func(k int, elemts []Elemt, space Space, src *rand.Rand) (centroids Clust, err error)

// Assign returns the element nearest centroid, its label and the distance to the centroid
func (c *Clust) Assign(elemt Elemt, space Space) (centroid Elemt, label int, dist float64) {
	label, dist = c.nearest(elemt, space)
	centroid = (*c)[label]
	return
}

// MapLabel assigns elements to centroids
func (c *Clust) MapLabel(elemts []Elemt, space Space) (labels []int, dists []float64) {
	labels = make([]int, len(elemts))
	dists = make([]float64, len(elemts))

	for i, elemt := range elemts {
		labels[i], dists[i] = c.nearest(elemt, space)
	}

	return
}

// ParMapLabel assigns elements to centroids in parallel
func (c *Clust) ParMapLabel(elemts []Elemt, space Space, degree int) (labels []int, dists []float64) {
	return parMapLabel(*c, elemts, space, degree)
}

// ReduceDBA computes centroids and cardinality of each clusters for given elements.
func (c *Clust) ReduceDBA(elemts []Elemt, space Space) (centroids Clust, cards []int) {
	centroids = make(Clust, len(*c))
	cards = make([]int, len(*c))

	for _, elemt := range elemts {
		var _, ix, _ = c.Assign(elemt, space)

		if cards[ix] == 0 {
			centroids[ix] = space.Copy(elemt)
			cards[ix] = 1
		} else {
			centroids[ix] = space.Combine(centroids[ix], cards[ix], elemt, 1)
			cards[ix]++
		}
	}

	return
}

// ReduceDBAForLabels computes loss and cardinality in each cluster for the given labels
func (c *Clust) ReduceDBAForLabels(elemts []Elemt, labels []int, space Space) (means []Elemt, cards []int) {
	means = make([]Elemt, len(*c))
	cards = make([]int, len(*c))
	for i, elemt := range elemts {

		var label = labels[i]

		if cards[label] == 0 {
			means[label] = space.Copy(elemt)
			cards[label] = 1
		} else {
			means[label] = space.Combine(means[label], cards[label], elemt, 1)
			cards[label]++
		}
	}
	return
}

// ParReduceDBAForLabels computes loss and cardinality in each cluster for the given labels in parallel
func (c *Clust) ParReduceDBAForLabels(elemts []Elemt, labels []int, space Space, degree int) ([]Elemt, []int) {
	return parDBAForLabels(*c, elemts, labels, space, degree)
}

// ParReduceDBA computes centroids and cardinality of each clusters for given elements in parallel.
func (c *Clust) ParReduceDBA(elemts []Elemt, space Space, degree int) (Clust, []int) {
	return parReduceDBA(*c, elemts, space, degree)
}

// TotalLoss computes loss from distances between elements and their nearest centroid
func (c *Clust) TotalLoss(elemts []Elemt, space Space, norm float64) float64 {
	losses, _ := c.ReduceLoss(elemts, space, norm)
	return floats.Sum(losses)
}

// ParTotalLoss computes loss from distances between elements and their nearest centroid in parallel
func (c *Clust) ParTotalLoss(elemts []Elemt, space Space, norm float64, degree int) float64 {
	losses, _ := c.ParReduceLoss(elemts, space, norm, degree)
	return floats.Sum(losses)
}

// ReduceLoss computes loss and cardinality in each cluster for the given elements
func (c *Clust) ReduceLoss(elemts []Elemt, space Space, norm float64) ([]float64, []int) {
	var losses = make([]float64, len(*c))
	var cards = make([]int, len(*c))
	for _, elemt := range elemts {
		var label, min = c.nearest(elemt, space)
		cards[label]++
		losses[label] += math.Pow(min, norm)
	}
	return losses, cards
}

// ParReduceLoss computes loss and cardinality in each cluster for the given elements in parallel
func (c *Clust) ParReduceLoss(elemts []Elemt, space Space, norm float64, degree int) ([]float64, []int) {
	return parLoss(*c, elemts, space, norm, degree)
}

// ReduceLossForLabels computes loss and cardinality in each cluster for the given labels
func (c *Clust) ReduceLossForLabels(elemts []Elemt, labels []int, space Space, norm float64) ([]float64, []int) {
	var losses = make([]float64, len(*c))
	var cards = make([]int, len(*c))
	for i := range elemts {
		var label = labels[i]
		var dist = space.Dist(elemts[i], (*c)[label])
		cards[label]++
		losses[label] += math.Pow(dist, norm)
	}
	return losses, cards
}

// ParReduceLossForLabels computes loss and cardinality in each cluster for the given labels in parallel
func (c *Clust) ParReduceLossForLabels(elemts []Elemt, labels []int, space Space, norm float64, degree int) ([]float64, []int) {
	return parLossForLabels(*c, elemts, labels, space, norm, degree)
}

// nearest Returns the label of element nearest centroid and the distance
func (c *Clust) nearest(elemt Elemt, space Space) (label int, min float64) {
	min = space.Dist(elemt, (*c)[0])
	label = 0

	for i := 1; i < len(*c); i++ {
		if d := space.Dist(elemt, (*c)[i]); min > d {
			min = d
			label = i
		}
	}

	return label, min
}

// DBA returns the averaged element
func DBA(elemts []Elemt, space Space) (dba Elemt, err error) {

	if len(elemts) == 0 {
		err = errors.New("DBA needs at least one elements")
		return
	}

	dba = space.Copy(elemts[0])
	var weight = 1

	for i := 1; i < len(elemts); i++ {
		dba = space.Combine(dba, weight, elemts[i], 1)
		weight++
	}

	return
}

// WeightedDBA computes the average of given elements with given weights
func WeightedDBA(elemts []Elemt, weights []int, space Space) (dba Elemt, err error) {

	if len(elemts) == 0 {
		err = errors.New("DBA needs at least one elements")
		return
	}

	dba = space.Copy(elemts[0])
	var weight = weights[0]

	for i := 1; i < len(elemts); i++ {
		dba = space.Combine(dba, weight, elemts[i], weights[i])
		weight += weights[i]
	}

	return
}

// Initializer that always returns the centroids
func (c *Clust) Initializer(int, []Elemt, Space, *rand.Rand) (centroids Clust, err error) {
	return *c, nil
}
