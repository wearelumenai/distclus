package clustering_go

import (
	"math/rand"
)

// Online clustering algorithm interface.
type OnlineClust interface {
	// Return the cluster information with the index idx.
	get(idx int) Cluster
	// Add an element to clustering data set.
	push(elemt Elemt)
	// Make a prediction on a element and return the cluster index.
	predict(elemt Elemt) int
	// Run clustering algorithm.
	run()
	// Close algorithm clustering process.
	close()
}

type Cluster struct {
	center Elemt
	elemts []Elemt
}

// Randomly picks initial K centroid nodes.
func randomInit(nClusters int, elemts []Elemt) []Elemt {
	centroids := make([]Elemt, nClusters)
	choices := make([]int, 0)
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
	return centroids
}

// Returns the index of the closest element to centroid from a list of nodes.
func assign(centroid Elemt, elemts []Elemt, space space) int {
	distances := make([]float64, len(elemts))
	for i, node := range elemts {
		distances[i] = space.dist(centroid, node)
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

// Compute a kmeans on nodes contained in space with nCluster centroids.
// Centroids are initialized through initializer method that return centroids(panic if collection is empty).
func kMeans(elemts []Elemt, space space, nClusters int, maxIter int,
	initializer func(k int, nodes []Elemt) []Elemt) map[int][]Elemt {
	centroids := initializer(nClusters, elemts)
	if len(centroids) == 0 {
		panic("no centroids return by initializer")
	}
	clusters := make(map[int][]Elemt)
	for iter := 0; iter < maxIter; iter++ {
		clusters = make(map[int][]Elemt)
		for _, node := range elemts {
			idxCluster := assign(node, centroids, space)
			clusters[idxCluster] = append(clusters[idxCluster], node)
		}
		for k, cluster := range clusters {
			centroids[k] = mean(cluster, space)
		}
	}
	return clusters
}
