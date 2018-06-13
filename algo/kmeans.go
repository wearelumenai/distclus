package algo

import (
	"errors"
	"fmt"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
)

type KMeansConf struct {
	K     int
	Iter  int
	Space core.Space
}

type KMeans struct {
	KMeansConf
	KMeansSupport
	Data        []core.Elemt
	status      ClustStatus
	initializer Initializer
	clust       Clust
	src         *rand.Rand
}

type KMeansSupport interface {
	Iterate(km KMeans) Clust
}

type SeqKMeansIterate struct {
}

func (SeqKMeansIterate) Iterate(km KMeans) Clust {
	var clust, _ = km.Centroids()
	var assign = clust.AssignAll(km.Data, km.Space)
	var result = make(Clust, len(clust))

	for k, cluster := range assign {
		if len(cluster) != 0 {
			result[k] = DBA(cluster, km.Space)
		}
	}

	return result
}

func NewKMeans(conf KMeansConf, initializer Initializer) KMeans {
	var km KMeans

	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}

	if conf.K < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.K))
	}

	km.KMeansConf = conf
	km.KMeansSupport = SeqKMeansIterate{}
	km.initializer = initializer
	km.status = Created
	km.src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))

	return km
}

func (km *KMeans) initialize() (error) {
	if len(km.Data) < km.K {
		return errors.New("can't initialize kmeans model centroids, not enough Data")
	}

	var clust = km.initializer(km.K, km.Data, km.Space, km.src)

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
	km.Data = append(km.Data, elemt)
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
		pred, idx, _ = km.clust.Assign(elemt, km.Space)
	}

	if push {
		km.Push(elemt)
	}

	return pred, idx, err
}

func (km *KMeans) Run() {
	km.status = Running
	km.initialize()
	for iter := 0; iter < km.Iter; iter++ {
		km.clust = km.Iterate(*km)
	}
}
