package clustering_go

import (
	"math/rand"
	"fmt"
)

type ClustStatus int

const (
	Created     ClustStatus = iota
	Initialized
	Running
	Closed
)

// Online clustering algorithm interface.
type OnlineClust interface {
	// Add an element to clustering data set.
	Push(elemt Elemt)
	// Return model current centroids configuration
	Centroids() (*[]Elemt, error)
	// Make a prediction on a element and return the cluster index.
	Predict(elemt Elemt) (Cluster, error)
	// Run clustering algorithm.
	Run()
	// Close algorithm clustering process.
	Close()
}

type Cluster struct {
	centroid Elemt
	elemts   []Elemt
}

type clustering struct {
	centroids []Elemt
	clusters  [][]Elemt
}

func (c *clustering) getCluster(idx int) Cluster {
	return Cluster{
		centroid: c.centroids[idx],
		elemts:   c.clusters[idx],
	}
}

func (c *clustering) getCentroids() *[]Elemt {
	return &c.centroids
}

func newClustering(centroids []Elemt, clusters [][]Elemt) (clustering, error) {
	var c clustering
	var clustlen = len(clusters)
	var centrolen = len(centroids)
	if clustlen < 1 {
		return c, fmt.Errorf("centroids collection is empty")
	}
	if centrolen < 1 {
		return c, fmt.Errorf("clustering collection is empty")
	}
	if clustlen != centrolen {
		return c, fmt.Errorf("clustering and centroids don't have the same dimension, %v != %v", clustlen, centrolen)
	}
	c = clustering{centroids: centroids, clusters: clusters}
	return c, nil
}

// Random initializer for clustering model.
func randomInit(nClusters int, elemts []Elemt, space space) clustering {
	var centroids = make([]Elemt, nClusters)
	var clusters = make([][]Elemt, nClusters)
	var choices = make([]int, 0)
	var i int
	for i < nClusters {
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
	for _, elemt := range elemts {
		var idx = assign(elemt, centroids, space)
		clusters[idx] = append(clusters[idx], elemt)
	}
	var c, _ = newClustering(centroids, clusters)
	return c
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
	mean := elemts[1]
	weight := 1
	for _, node := range elemts {
		mean = space.combine(node, 1, mean, weight)
		weight += 1
	}
	return mean
}
