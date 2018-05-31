package clustering_go

import (
	"fmt"
	"math"
	"math/rand"
)

type ClustStatus int

const (
	Created     ClustStatus = iota
	Initialized
	Running
	Closed
)

type Initializer = func(k int, nodes []Elemt, space space) (Clust, error)

// Online Clust algorithm interface.
type OnlineClust interface {
	// Add an element to Clust data set.
	Push(elemt Elemt)
	// Return model current centroids configuration.
	Centroids() (Clust, error)
	// Make a prediction on a element and return the associated center and its index.
	Predict(elemt Elemt) (Elemt, int, error)
	// Run Clust algorithm.
	Run()
	// Close algorithm Clust process.
	Close()
}

// Indexed clustering result
type Clust struct {
	centers []Elemt
}

// Return centers array pointer
func (c *Clust) Centers() *[]Elemt {
	return &c.centers
}

// Return center at the idx index
func (c *Clust) Center(idx int) Elemt {
	return c.centers[idx]
}

// Set centers
func (c *Clust) SetCenters(centers []Elemt) {
	c.centers = centers
}

// Assign elements on elemts at each centers
func (c* Clust) Assign(elemts *[]Elemt, space space) [][]Elemt {
	var clusters = make([][]Elemt, len(c.centers))
	for _, elemt := range *elemts {
		var idx = assign(elemt, c.centers, space)
		clusters[idx] = append(clusters[idx], elemt)
	}
	return clusters
}

// Compute loss of centers configuration with given data
func (c *Clust) Loss(data *[]Elemt, space space, norm float64) float64 {
	var sum float64
	for _, elemt := range *data {
		var min = math.MaxFloat64
		for _, center := range c.centers {
			min = math.Min(min, math.Pow(space.dist(elemt, center), norm))
		}
		sum += min
	}
	return sum / float64(len(*data))
}

// Clustering constructor
func NewClustering(centroids []Elemt) (Clust, error) {
	var c Clust
	if len(centroids) < 1 {
		return c, fmt.Errorf("centroids collection is empty")
	}
	return Clust{centroids}, nil
}

// Random initializer for Clust model.
func NewRandClustering(k int, elemts []Elemt) Clust {
	if len(elemts) < k {
		panic("not enough elements to initialize")
	}
	var centroids = make([]Elemt, k)
	var choices = make([]int, 0)
	var i int
	for i < k {
		choice := rand.Intn(len(elemts))
		find := false
		for _, v := range choices {
			if v == choice {
				find = true
			}
		}
		if !find {
			centroids[i] = elemts[choice]
			choices = append(choices, choice)
			i++
		}
	}
	return Clust{centroids}
}

// Returns the index of the closest element to elemt in elemts.
func assign(elemt Elemt, elemts []Elemt, space space) int {
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

// Return the mean of nodes based on the space combination method.
// If nodes are empty function panic.
func mean(elemts []Elemt, space space) Elemt {
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

// Random clustering initializer
func RandInitializer(k int, elemts []Elemt, _ space) (c Clust, err error) {
	if p := recover(); p != nil {
		return c, fmt.Errorf("%v", p)
	}
	return NewRandClustering(k, elemts), err
}