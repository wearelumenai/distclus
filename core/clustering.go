package core

import (
	"errors"
	"math"

	"golang.org/x/exp/rand"
)

// ClustStatus integer type
type ClustStatus = int64

// ClustStatus const values
const (
	Created ClustStatus = iota
	Ready
	Running
	Closed
)

// Clust type is an abbrevation for centroids indexed by labels.
type Clust []Elemt

// Initializer function initializes k centroids from the given elements.
type Initializer func(k int, elemts []Elemt, space Space, src *rand.Rand) (centroids Clust, err error)

// ReduceDBA returns centroids and cardinalities in each clusters.
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

func (c *Clust) ParReduceDBA(elemts []Elemt, space Space, degree int) (Clust, []int) {
	return parReduceDBA(*c, elemts, space, degree)
}

// MapLabel assignes elemts to each centroids
func (c *Clust) MapLabel(elemts []Elemt, space Space) (labels []int) {
	labels = make([]int, len(elemts))

	for i, elemt := range elemts {
		var label, _ = c.nearest(elemt, space)
		labels[i] = label
	}

	return
}

// MapLabel assignes elemts to each centroids
func (c *Clust) ParMapLabel(elemts []Elemt, space Space, degree int) (labels []int) {
	return parMapLabel(*c, elemts, space, degree)
}

// Assign returns the element nearest centroid, its label and the distance to the centroid
func (c *Clust) Assign(elemt Elemt, space Space) (centroid Elemt, label int, dist float64) {
	label, dist = c.nearest(elemt, space)
	centroid = (*c)[label]
	return
}

// TotalLoss computes loss from distances between elements and their nearest centroid
func (c *Clust) TotalLoss(elemts []Elemt, space Space, norm float64) float64 {
	losses, _ := c.ReduceLoss(elemts, space, norm)
	return sumLosses(losses)
}

func (c *Clust) ParTotalLoss(elemts []Elemt, space Space, norm float64, degree int) float64 {
	losses, _ := c.ParReduceLoss(elemts, space, norm, degree)
	return sumLosses(losses)
}

func sumLosses(losses []float64) float64 {
	var sum = 0.
	for _, loss := range losses {
		sum += loss
	}
	return sum
}

func (c *Clust) ReduceLoss(elemts []Elemt, space Space, norm float64) ([]float64, []int) {
	var losses = make([]float64, len(*c))
	var cards = make([]int, len(*c))
	for _, elemt := range elemts {
		var label, min = c.nearest(elemt, space)
		cards[label] += 1
		losses[label] += math.Pow(min, norm)
	}
	return losses, cards
}

func (c *Clust) ParReduceLoss(elemts []Elemt, space Space, norm float64, degree int) ([]float64, []int) {
	return ParLosses(*c, elemts, space, norm, degree)
}

// Returns the label of element nearest centroid and the distance
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
