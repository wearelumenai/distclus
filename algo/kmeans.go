package algo

import (
	"fmt"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
	"errors"
)

type KMeansConf struct {
	K     int
	Iter  int
	Space core.Space
	RGen  *rand.Rand
}

type KMeans struct {
	KMeansConf
	KMeansSupport
	Data        []core.Elemt
	status      ClustStatus
	initializer Initializer
	clust       Clust
	rgen        *rand.Rand
	closing     chan bool
	closed      chan bool
}

type KMeansSupport interface {
	Iterate(km KMeans, proposal Clust) Clust
}

type SeqKMeansIterate struct {
}

func (SeqKMeansIterate) Iterate(km KMeans, clust Clust) Clust {
	var assign = clust.AssignAll(km.Data, km.Space)
	var result = make(Clust, len(clust))

	for k, cluster := range assign {
		if len(cluster) != 0 {
			result[k], _ = DBA(cluster, km.Space)
		} else {
			result[k] = clust[k]
		}
	}

	return result
}

func NewKMeans(conf KMeansConf, initializer Initializer) KMeans {

	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}

	var km KMeans
	km.KMeansConf = conf
	km.KMeansSupport = SeqKMeansIterate{}
	km.initializer = initializer
	km.status = Created

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		km.rgen = rand.New(rand.NewSource(seed))
	} else {
		km.rgen = conf.RGen
	}

	km.closing = make(chan bool, 1)
	km.closed = make(chan bool, 1)

	return km
}

func (km *KMeans) Centroids() (c Clust, err error) {
	switch km.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = km.clust
	}

	return
}

func (km *KMeans) Push(elemt core.Elemt) (err error) {
	switch km.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		km.Data = append(km.Data, elemt)
	}

	return err
}

func (km *KMeans) Close() {
	km.closing <- true
	<-km.closed
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int

	var clust, err = km.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, km.Space)
		if push {
			err = km.Push(elemt)
		}
	}

	return pred, idx, err
}

func (km *KMeans) Run(async bool) {
	km.status = Running
	km.clust = km.initializer(km.KMeansConf.K, km.Data, km.KMeansConf.Space, km.rgen)

	var do = func() {
		for iter, loop := 0, true; iter < km.Iter && loop; iter++ {
			select {
			case <-km.closing:
				loop = false
			default:
				km.clust = km.Iterate(*km, km.clust)
			}
		}

		km.status = Closed
		km.closed <- true
	}

	if async {
		go do()
	} else {
		do()
	}
}
