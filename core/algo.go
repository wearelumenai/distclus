// Package core proposes a generic framework that executes online clustering algorithm.
package core

import (
	"errors"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
)

// StatusNotifier for being notified by Online clustering change status
type StatusNotifier = func(ClustStatus)

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
	Run() error
	RunOC(StatusNotifier) error
	Pause() error
	WakeUp() error
	Close() error
	RuntimeFigures() (map[string]float64, error)
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
	runtimeFigures map[string]float64
	statusNotifier StatusNotifier
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

func (algo *Algo) setStatus(status ClustStatus) {
	atomic.StoreInt64(&algo.status, status)
	if algo.statusNotifier != nil {
		algo.statusNotifier(status)
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
		centroids = algo.centroids
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

// Run executes the algorithm in batch mode
func (algo *Algo) Run() (err error) {
	err = algo.tryInit()
	if err == nil {
		err = algo.runIfReady(false)
	}
	return
}

// ARun executes asynchronously the algorithm in Online Clustering mode
func (algo *Algo) RunOC(notifier StatusNotifier) (err error) {
	algo.statusNotifier = notifier
	err = algo.tryInit()
	if err == nil {
		err = algo.runIfReady(true)
	}
	return
}

// Space returns space
func (algo *Algo) Space() Space {
	return algo.space
}

// Pause algorithm execution
func (algo *Algo) Pause() (err error) {
	if algo.Status() == Running {
		err = algo.impl.Pause()
	} else {
		err = fmt.Errorf("Algorithm is not running")
	}
	return
}

// WakeUp algorithm execution after paused status
func (algo *Algo) WakeUp() (err error) {
	if algo.Status() == Paused {
		err = algo.impl.WakeUp()
	} else {
		err = fmt.Errorf("Algorithm is not paused")
	}
	return
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
	if algo.status == Running || algo.status == Paused {
		algo.closing <- true
		<-algo.closed
	}
	algo.setStatus(Closed)
	return
}

// RuntimeFigures returns specific algo properties
func (algo *Algo) RuntimeFigures() (figures map[string]float64, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not running")
	default:
		algo.mutex.RLock()
		defer algo.mutex.RUnlock()
		figures = algo.runtimeFigures
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

// Status returns the status of the algorithm
func (algo *Algo) Status() ClustStatus {
	var status = atomic.LoadInt64(&algo.status)
	return status
}

func (algo *Algo) tryInit() (err error) {
	if algo.status == Created {
		algo.centroids, err = algo.impl.Init(algo.conf.ImplConf, algo.space)
		if err == nil {
			algo.setStatus(Ready)
		}
	}
	return
}

func (algo *Algo) runIfReady(async bool) (err error) {
	if algo.status == Ready {
		err = algo.run(async)
	} else {
		err = fmt.Errorf("invalid status %v", algo.status)
	}
	return
}

func (algo *Algo) run(async bool) (err error) {
	if async {
		err = algo.impl.SetAsync(algo.setStatus)
		if err == nil {
			go algo.runAsync()
		}
	} else {
		err = algo.runSync()
	}
	return
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) runSync() (err error) {
	algo.setStatus(Running)

	err = algo.impl.Run(
		algo.conf.ImplConf,
		algo.space,
		algo.centroids,
		algo.updateCentroids,
		algo.closing,
		algo.closed,
	)

	if algo.status == Running {
		algo.setStatus(Ready)
	}

	return
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) runAsync() {
	var err error
	for algo.status == Ready {
		err = algo.runSync()
		if err != nil {
			log.Println(err)
		}
	}
}

func (algo *Algo) updateCentroids(centroids Clust, figures map[string]float64) {
	algo.mutex.Lock()
	defer algo.mutex.Unlock()
	algo.centroids = centroids
	algo.runtimeFigures = figures
}

// ErrTimeOut is returned when an error occurs
var ErrTimeOut = errors.New("algorithm timed out")
