package algo

import (
	"errors"
	"fmt"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
)

type KMeans struct {
	iter        int
	k           int
	data        []core.Elemt
	space       core.Space
	status      ClustStatus
	initializer Initializer
	clust       Clust
	src         *rand.Rand
}

func NewKMeans(k int, iter int, space core.Space, initializer Initializer) KMeans {
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
	km.src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))

	return km
}

func (km *KMeans) initialize() (error) {
	if len(km.data) < km.k {
		return errors.New("can't initialize kmeans model centroids, not enough data")
	}

	var clust = km.initializer(km.k, km.data, km.space, km.src)

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

func (km *KMeans) Push(elemt core.Elemt) {
	km.data = append(km.data, elemt)
}

func (km *KMeans) Close() {
	km.status = Closed
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int
	var err error

	switch km.status {
	case Created:
		err = fmt.Errorf("no Clust available")
	default:
		pred, idx = km.clust.UAssign(elemt, km.space)
	}

	if push {
		km.Push(elemt)
	}

	return pred, idx, err
}

func (km *KMeans) iteration() {
	var clusters = km.clust.Assign(km.data, km.space)
	for k, cluster := range clusters {
		if len(cluster) != 0 {
			km.clust[k] = mean(cluster, km.space)
		}
	}
}

func (km *KMeans) Run() {
	km.status = Running
	km.initialize()
	for iter := 0; iter < km.iter; iter++ {
		km.iteration()
	}
}