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
	initializer func(k int, nodes []Elemt) clustering
	clustering  clustering
}

func NewKMeans(k int, iter int, space space, initializer func(k int, elemts []Elemt, space space) clustering) KMeans {
	var km KMeans
	if k < 1 {
		panic(fmt.Sprintf("Illegal value for k: %v", k))
	}
	if k < 0 {
		panic(fmt.Sprintf("Illegal value for iter: %v", k))
	}
	km.iter = iter
	km.k = k
	km.initializer = func(k int, elemts []Elemt) clustering {
		return initializer(k, elemts, km.space)
	}
	km.space = space
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

func (km *KMeans) Centroids() (*[]Elemt, error) {
	var c []Elemt
	var err error
	switch km.status {
	case Created:
		err = fmt.Errorf("no clustering available")
	default:
		c = km.clustering.getAllCenter()
	}
	return &c, err
}

func (km *KMeans) Push(elemt Elemt) {
	km.data = append(km.data, elemt)
}

func (km *KMeans) Close() {
	km.status = Closed
}

func (km *KMeans) Predict(elemt Elemt) (*Cluster, error) {
	var c *Cluster
	switch km.status {
	case Created:
		return c, fmt.Errorf("no clustering available")
	default:
		var idx = assign(elemt, km.clustering.getAllCenter(), km.space)
		c, err := km.clustering.getClust(idx)
		if err != nil {
			panic(err)
		}
		return c, nil
	}
}

func (km *KMeans) iteration() {
	var clusters = make([][]Elemt, km.k)
	var centroids = km.clustering.getAllCenter()
	for _, node := range km.data {
		var idxCluster = assign(node, centroids, km.space)
		clusters[idxCluster] = append(clusters[idxCluster], node)
	}
	for k, cluster := range clusters {
		centroids[k] = mean(cluster, km.space)
	}
	var clustering, err = newClustering(centroids, clusters)
	if err != nil {
		panic(err)
	}
	km.clustering = clustering
}

func (km *KMeans) Run() {
	km.status = Running
	km.initialize()
	for iter := 0; iter < km.iter; iter++ {
		km.iteration()
	}
}
