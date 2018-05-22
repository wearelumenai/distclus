package clustering_go

import (
	"math/rand"
)

// Online clustering algorithm interface.
type OnlineClust interface {
	// Return the cluster information with the index idx.
	get(idx int) (node, []node)
	// Add an element to clustering data set.
	push(node node)
	// Make a prediction on node and return the cluster index.
	predict(node node) int
	// Run clustering algorithm.
	run()
	// Close algorithm clustering process.
	close()
}

// Randomly picks initial K centroid nodes.
func randomInit(nClusters int, nodes []node) []node {
	centroids := make([]node, nClusters)
	choices := make([]int, 0)
	var i int
	for i < nClusters {
		choice := rand.Intn(len(nodes))
		find := false
		for _, v := range choices {
			if v == choice {
				find = true
			}
		}
		if !find {
			centroids[i] = nodes[choice]
			choices = append(choices, choice)
			i++
		}
	}
	return centroids
}

// Returns the index of the closest node to centroid from a list of nodes.
func assign(centroid node, nodes []node, space space) int {
	distances := make([]float64, len(nodes))
	for i, node := range nodes {
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
func mean(nodes []node, space space) node {
	l := len(nodes)
	if l < 1 {
		panic("nodes are empty")
	}
	mean := nodes[1]
	weight := 1
	for _, node := range (nodes) {
		mean = space.combine(node, 1, mean, weight)
		weight += 1
	}
	return mean
}

// Compute a kmeans on nodes contained in space with nCluster centroids.
// Centroids are initialized through initializer method that return centroids(panic if collection is empty).
func kMeans(nodes []node, space space, nClusters int, maxIter int,
	initializer func(k int, nodes []node) []node) map[int][]node {
	centroids := initializer(nClusters, nodes)
	if len(centroids) == 0 {
		panic("no centroids return by initializer")
	}
	clusters := make(map[int][]node)
	for iter := 0; iter < maxIter; iter++ {
		clusters = make(map[int][]node)
		for _, node := range nodes {
			idxCluster := assign(node, centroids, space)
			clusters[idxCluster] = append(clusters[idxCluster], node)
		}
		for k, cluster := range clusters {
			centroids[k] = mean(cluster, space)
		}
	}
	return clusters
}
