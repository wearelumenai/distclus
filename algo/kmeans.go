package algo

import (
	"fmt"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
	"errors"
)

type KMeansSupport interface {
	Iterate(km KMeans, proposal core.Clust) core.Clust
}

type SeqKMeansSupport struct {
}

func (SeqKMeansSupport) Iterate(km KMeans, clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))
	var cards = make([]int, len(clust))

	for i, _ := range km.Data {
		var _, ix, _ = clust.Assign(km.Data[i], km.Space)

		if cards[ix] == 0 {
			result[ix] = km.Space.Copy(km.Data[i])
			cards[ix] = 1
		} else {
			km.Space.Combine(result[ix], cards[ix], km.Data[i], 1)
			cards[ix] += 1
		}
	}

	for i := 0; i < len(result); i++ {
		if result[i] == nil {
			result[i] = clust[i]
		}
	}

	return result
}

type KMeansConf struct {
	K     int
	Iter  int
	Space core.Space
	RGen  *rand.Rand
}

type KMeans struct {
	KMeansConf
	KMeansSupport
	Buffer
	status      core.ClustStatus
	initializer Initializer
	clust       core.Clust
	rgen        *rand.Rand
	closing     chan bool
	closed      chan bool
}

func NewKMeans(conf KMeansConf, initializer Initializer, data []core.Elemt) KMeans {

	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}

	var km KMeans
	km.KMeansConf = conf
	km.KMeansSupport = SeqKMeansSupport{}
	km.initializer = initializer
	km.status = core.Created

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		km.rgen = rand.New(rand.NewSource(seed))
	} else {
		km.rgen = conf.RGen
	}

	km.closing = make(chan bool, 1)
	km.closed = make(chan bool, 1)

	if data == nil {
		data = make([]core.Elemt, 0)
	}
	km.Buffer = newBuffer(data, -1)

	return km
}

func (km *KMeans) Centroids() (c core.Clust, err error) {
	switch km.status {
	case core.Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = km.clust
	}

	return
}

func (km *KMeans) Push(elemt core.Elemt) (err error) {
	switch km.status {
	case core.Closed:
		err = errors.New("clustering ended")
	default:
		km.Buffer.push(elemt)
	}

	return err
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

	if async {
		km.setAsync()

		go func() {
			for ok := false; !ok; {
				km.clust, ok = km.initializer(km.KMeansConf.K, km.Data, km.KMeansConf.Space, km.rgen)
				if !ok {
					time.Sleep(300 * time.Millisecond)
					km.Buffer.apply()
				}
			}

			km.status = core.Running
			km.process()
		}()

		time.Sleep(300 * time.Millisecond)
	} else {
		var ok bool
		km.clust, ok = km.initializer(km.KMeansConf.K, km.Data, km.KMeansConf.Space, km.rgen)

		if !ok {
			panic("failed to initialize")
		}

		km.status = core.Running
		km.process()
	}
}

func (km *KMeans) process() {
	for iter, loop := 0, true; iter < km.Iter && loop; iter++ {
		select {

		case <-km.closing:
			loop = false

		default:
			km.clust = km.Iterate(*km, km.clust)
			km.apply()
		}
	}

	km.status = core.Closed
	km.closed <- true
}

func (km *KMeans) Close() {
	km.closing <- true
	<-km.closed
}
