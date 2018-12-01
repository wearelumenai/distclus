package core

import (
	"errors"
	"fmt"
	"sync"
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
	ARun(async bool, notifier Notifier) error
	Close() error
}

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	Conf           Conf
	Impl           Impl
	Space          Space
	centroids      Clust
	status         ClustStatus
	closing        chan bool
	closed         chan bool
	mutex          sync.Mutex
	lastUpdateTime int64
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
		centroids = algo.centroids.Copy()
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

// ARun executes the algorithm and notify the user with a callback, timed by a time to callback (ttc) integer
func (algo *Algo) ARun(async bool, notifier Notifier) (err error) {
	if async {
		err = algo.Impl.SetAsync()
		if err == nil {
			go algo.initAndRunAsync(notifier)
		}
	} else {
		err = algo.initAndRunSync(notifier)
	}
	if err == nil {
		algo.status = Running
	}

	return
}

// Run the algorithm, asynchronously if async is true
func (algo *Algo) Run(async bool) (err error) {
	return algo.ARun(async, Notifier{})
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

func (algo *Algo) updateCentroids(notifier Notifier) func(Clust) {
	return func(centroids Clust) {
		algo.centroids = centroids
		if notifier.Callback != nil && (algo.lastUpdateTime == 0 || (notifier.TTC <= time.Now().UnixNano()-int64(algo.lastUpdateTime))) {
			algo.lastUpdateTime = time.Now().UnixNano()
			notifier.Callback(centroids)
		}
	}
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) initAndRunSync(notifier Notifier) (err error) {
	algo.centroids, err = algo.Impl.Init(algo.Conf, algo.Space)
	if err == nil {
		// go algo.updateCentroids(callback, cchan, ttc)
		err = algo.Impl.Run(
			algo.Conf,
			algo.Space,
			algo.centroids,
			algo.updateCentroids(notifier),
			algo.closing,
		)
		if err == nil {
			algo.closed <- true
		}
	}

	return
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) initAndRunAsync(notifier Notifier) (err error) {
	if err = algo.initAndRunSync(notifier); err != nil {
		time.Sleep(300 * time.Millisecond)
		err = algo.initAndRunAsync(notifier)
	}
	return
}
