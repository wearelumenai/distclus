package core

import (
	"errors"
	"fmt"
	"sync"
)

// OnlineClust interface
// When a prediction is made, the element can be pushed to the model.
// A prediction consists in a centroid and a label.
// The following constraints must be met (otherwise an error is returned) :
// an element can't be pushed if the algorithm is closed,
// a prediction can't be done before the algorithm is run,
// no centroid can be returned before the algorithm is run.
type OnlineClust interface {
	Centroids() (Clust, error)
	Push(elemt Elemt) error
	Predict(elemt Elemt) (Elemt, int, error)
	Run(async bool) error
	Close() error
	RuntimeFigure(name string) (float64, error)
}

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	conf           Conf
	impl           Impl
	space          Space
	centroids      Clust
	status         ClustStatus
	closing        chan bool
	closed         chan bool
	lastUpdateTime int64
	mutex          sync.RWMutex
}

// AlgoConf algorithm configuration
type AlgoConf interface{}

// NewAlgo creates a new algorithm instance
func NewAlgo(conf Conf, impl Impl, space Space) Algo {
	return Algo{
		conf:    conf,
		impl:    impl,
		space:   space,
		status:  Created,
		closing: make(chan bool, 1),
		closed:  make(chan bool, 1),
	}
}

// Centroids Get the centroids currently found by the algorithm
func (algo *Algo) Centroids() (centroids Clust, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		algo.mutex.RLock()
		defer algo.mutex.RUnlock()
		var algoCentroids = algo.centroids
		centroids = make(Clust, len(algoCentroids))
		for index, centroid := range algoCentroids {
			centroids[index] = algo.space.Copy(centroid)
		}
	}
	return
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		err = algo.impl.Push(elemt)
	}
	return
}

// Run executes the algorithm and notify the user with a callback, timed by a time to callback (ttc) integer
func (algo *Algo) Run(async bool) (err error) {
	err = algo.tryInit()
	if err == nil {
		err = algo.runIfReady(async)
	}
	return
}

func (algo *Algo) tryInit() (err error) {
	if algo.status == Created {
		algo.centroids, err = algo.impl.Init(algo.conf.ImplConf, algo.space)
		if err == nil {
			algo.status = Ready
		}
	}
	return
}

func (algo *Algo) runIfReady(async bool) (err error) {
	if algo.status == Ready {
		if async {
			err = algo.impl.SetAsync()
			if err == nil {
				algo.status = Running
				go algo.runAsync()
			}
		} else {
			algo.status = Running
			err = algo.runSync()
			algo.status = Ready
		}
	} else {
		err = fmt.Errorf("invalid status %v", algo.status)
	}
	return
}

// Conf returns configuration
func (algo *Algo) Conf() Conf {
	return algo.conf
}

// Impl returns impl
func (algo *Algo) Impl() Impl {
	return algo.impl
}

// Space returns space
func (algo *Algo) Space() Space {
	return algo.space
}

// Predict the cluster for a new observation
func (algo *Algo) Predict(elemt Elemt) (pred Elemt, label int, err error) {
	var clust Clust
	clust, err = algo.Centroids()

	if err == nil {
		pred, label, _ = clust.Assign(elemt, algo.space)
	}

	return
}

// Close Stops the algorithm
func (algo *Algo) Close() (err error) {
	if algo.status == Running {
		algo.closing <- true
		<-algo.closed
	}
	algo.status = Closed
	return
}

func (algo *Algo) updateCentroids(centroids Clust) {
	algo.mutex.Lock()
	defer algo.mutex.Unlock()
	algo.centroids = centroids
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) runSync() (err error) {
	return algo.impl.Run(
		algo.conf.ImplConf,
		algo.space,
		algo.centroids,
		algo.updateCentroids,
		algo.closing,
		algo.closed,
	)
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) runAsync() {
	var err error
	for algo.status != Closed {
		err = algo.runSync()
		if err != nil {
			fmt.Print(err)
		}
	}
}

// Iterations returns number of iterations per execution
func (algo *Algo) RuntimeFigure(name string) (figure float64, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not running")
	default:
		figure, err = algo.impl.RuntimeFigure(name)
	}
	return
}
