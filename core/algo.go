package core

import (
	"errors"
	"fmt"
	"time"
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
	Predict(elemt Elemt, push bool) (Elemt, int, error)
	Run(async bool) error
	Close() error
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
		centroids, err = algo.Impl.Centroids()
	}
	return
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		err = algo.Impl.Push(elemt)
	}
	return
}

// Run the algorithm, asynchronously if async is true
func (algo *Algo) Run(async bool) (err error) {
	if async {
		err = algo.Impl.SetAsync()
		if err == nil {
			go algo.initAndRunAsync()
		}
	} else {
		err = algo.initAndRunSync()
	}
	if err == nil {
		algo.status = Running
	}

	return
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
func (algo *Algo) Close() (err error) {
	if algo.status == Running {
		algo.closing <- true
		<-algo.closed
	}
	algo.status = Closed
	return
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) initAndRunSync() (err error) {
	err = algo.Impl.Init(algo.Conf, algo.Space)
	if err == nil {
		err = algo.Impl.Run(algo.Conf, algo.Space, algo.closing)
		if err == nil {
			algo.closed <- true
		}
	}

	return
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) initAndRunAsync() (err error) {
	if err = algo.initAndRunSync(); err != nil {
		time.Sleep(300 * time.Millisecond)
		err = algo.initAndRunAsync()
	}
	return
}
