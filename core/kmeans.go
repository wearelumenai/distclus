package core

import (
	"errors"
	"fmt"
	"time"
	"golang.org/x/exp/rand"
)

type KMeans struct {
	iter        int
	k           int
	data        []Elemt
	space       Space
	status      ClustStatus
	initializer Initializer
	clust       Clust
}

func NewKMeans(k int, iter int, space Space, initializer Initializer) KMeans {
	var km KMeans
	if k < 1 {
		panic(fmt.Sprintf("Illegal value for k: %v", k))
	}
	if k < 0 {
		panic(fmt.Sprintf("Illegal value for iter: %v", k))
	}
	km.iter = iter
	km.k = k
	km.initializer = initializer
	km.space = space
	km.status = Created
	return km
}

func (km *KMeans) initialize() (error) {
	if len(km.data) < km.k {
		return errors.New("can't initialize kmeans model centroids, not enough data")
	}
	var clust, err = km.initializer(km.k, km.data, km.space)
	if err != nil {
		panic(err)
	}
	km.clust = clust
	km.status = Initialized
	return nil
}

func (km *KMeans) Centroids() (c Clust, err error) {
	switch km.status {
	case Created:
		err = fmt.Errorf("no Clust available")
	default:
		c = km.clust
	}
	return c, err
}

func (km *KMeans) Push(elemt Elemt) {
	km.data = append(km.data, elemt)
}

func (km *KMeans) Close() {
	km.status = Closed
}

func (km *KMeans) Predict(elemt Elemt) (c Elemt, idx int, err error) {
	switch km.status {
	case Created:
		return c, idx, fmt.Errorf("no Clust available")
	default:
		c, idx = km.clust.UAssign(elemt, km.space)
		return c, idx, nil
	}
}

func (km *KMeans) iteration() {
	var clusters = km.clust.Assign(&km.data, km.space)
	var centroids = km.clust.centers
	for k, cluster := range clusters {
		if len(cluster) != 0{
			centroids[k] = mean(cluster, km.space)
		}
	}
	var clustering, err = NewClustering(centroids)
	if err != nil {
		panic(err)
	}
	km.clust = clustering
}

func (km *KMeans) Run() {
	km.status = Running
	km.initialize()
	for iter := 0; iter < km.iter; iter++ {
		km.iteration()
	}
}

// Run au kmeans++ on a batch to return a k centers configuration
func KmeansPP(k int, batch *[]Elemt, space Space, src *rand.Rand) (c Clust, err error) {
	if k < 1 {
		panic("k is lower than 1")
	}
	var init = make([]Elemt, 1)
	init[0] = (*batch)[src.Intn(len(*batch))]
	c = Clust{init}
	for i:=1; i < k; i++{
		c, err = KmeansPPIter(c, batch, space, src)
		if err!= nil {
			return c, err
		}
	}
	return c, nil
}

// Run au kmeans++ iteration on a batch to return a k+1 centers configuration
func KmeansPPIter(clust Clust, batch *[]Elemt, space Space, src *rand.Rand) (Clust, error) {
	l := len(*batch)
	var dists = make([]float64, l)
	for i, elt := range *batch {
		var center, _= clust.UAssign(elt, space)
		dists[i] = space.dist(elt, center)
	}
	return NewClustering(append(*clust.Centers(), (*batch)[WeightedChoice(dists, src)]))
}

// Kmeans++ clustering initializer
func KmeansPPInitializer(k int, elemts []Elemt, space Space) (c Clust, err error) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	return KmeansPP(k, &elemts, space, src)
}