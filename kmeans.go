package clustering_go

import (
	"fmt"
	"errors"
)

type KMeans struct {
	iter        int
	k           int
	data        []Elemt
	space       space
	status      ClustStatus
	initializer func(k int, nodes []Elemt) Clustering
	clustering  Clustering
}

func NewKMeans(k int, iter int, data []Elemt, space space, initializer func(k int, elemts []Elemt, space space) Clustering) KMeans {
	var km KMeans
	if k < 1 {
		panic(fmt.Sprintf("Illegal value for k: %v", k))
	}
	if k < 0 {
		panic(fmt.Sprintf("Illegal value for iter: %v", k))
	}
	km.iter = iter
	km.k = k
	km.initializer = func(k int, elemts []Elemt) Clustering {
		return initializer(k, elemts, km.space)
	}
	km.space = space
	km.data = data
	km.status = Created
	return km
}

func (km *KMeans) initialize() (error) {
	if len(km.data) < km.k {
		return errors.New("can't initialize kmeans model centroids, not enough data")
	}
	km.clustering = km.initializer(km.k, km.data)
	km.status = Initialized
	return nil
}

func (km *KMeans) configuration() (Clustering, error) {
	var c Clustering
	if km.status == Created {
		return c, fmt.Errorf("no clustering available")
	}
	c = km.clustering
	return c, nil
}

func (km *KMeans) push(elemt Elemt) {
	km.data = append(km.data, elemt)
}

func (km *KMeans) close() {
	km.status = Closed
}

func (km *KMeans) predict(elemt Elemt) Cluster {
	var idx = assign(elemt, *km.clustering.getCentroids(), km.space)
	return km.clustering.getCluster(idx)
}

func (km *KMeans) iteration() {
	var clusters = make([][]Elemt, km.k)
	var centroids = *km.clustering.getCentroids()
	for _, node := range km.data {
		var idxCluster = assign(node, centroids, km.space)
		clusters[idxCluster] = append(clusters[idxCluster], node)
	}
	for k, cluster := range clusters {
		centroids[k] = mean(cluster, km.space)
	}
	var clustering, err = NewClustering(centroids, clusters)
	if err != nil {
		panic(err)
	}
	km.clustering = clustering
}

func (km *KMeans) run() {
	km.status = Running
	km.initialize()
	for iter := 0; iter < km.iter; iter++ {
		km.iteration()
	}
}
