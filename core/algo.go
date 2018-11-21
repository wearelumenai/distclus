package core

import (
	"errors"
	"fmt"
	"time"
)

// Impl concrete algorithms
type Impl interface {
	Init(conf Conf) bool
	Run(conf Conf, space Space, closing <-chan bool)
	Push(elemt Elemt)
	SetAsync()
	Centroids() Clust
}

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	Conf    Conf
	Impl    Impl
	Space   Space
	status  ClustStatus
	closing chan bool
	closed  chan bool
}

// NewAlgo creates a new algorithm instance
func NewAlgo(conf Conf, impl Impl, space Space) Algo {
	return Algo{
		Conf:    conf,
		Impl:    impl,
		Space:   space,
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
		centroids = algo.Impl.Centroids()
	}
	return
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		algo.Impl.Push(elemt)
	}
	return
}

// Run the algorithm, asynchronously if async is true
func (algo *Algo) Run(async bool) {
	if async {
		algo.Impl.SetAsync()
		go algo.initAndRunAsync()
	} else {
		if err := algo.initAndRunSync(); err != nil {
			panic(err)
		}
	}
	algo.status = Running
}

// Local configuration for type casting
type spaceConf struct {
	Space
}

// Predict the cluster for a new observation
func (algo *Algo) Predict(elemt Elemt, push bool) (pred Elemt, label int, err error) {
	var clust Clust
	clust, err = algo.Centroids()

	if err == nil {
		pred, label, _ = clust.Assign(elemt, algo.Space)
		if push {
			err = algo.Push(elemt)
		}
	}

	return
}

// Close Stops the algorithm
func (algo *Algo) Close() {
	if algo.status == Running {
		algo.closing <- true
		<-algo.closed
	}
	algo.status = Closed
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) initAndRunSync() error {
	var ok = algo.Impl.Init(algo.Conf)
	if ok {
		algo.Impl.Run(algo.Conf, algo.Space, algo.closing)
		algo.closed <- true
		return nil
	}

	return errors.New("Failed to initialize")
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) initAndRunAsync() (err error) {
	if err = algo.initAndRunSync(); err != nil {
		time.Sleep(300 * time.Millisecond)
		err = algo.initAndRunAsync()
	}
	return
}
