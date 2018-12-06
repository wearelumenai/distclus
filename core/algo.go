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
	ARun(async bool, notifier Notifier) error
	Close() error
	SetConf(Conf) error
	SetSpace(Space) error
	Conf() Conf
	Space() Space
	Impl() Impl
	Reset(Conf, []Elemt) error
	Fit([]Elemt) error
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
}

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

// Fit applies fit algorithm execution
func (algo *Algo) Fit(data []Elemt) (err error) {
	err = algo.Reset(algo.conf, data)
	if err == nil {
		err = algo.Run(false)
	}
	if err == nil {
		err = algo.Close()
	}
	return
}

// Reset implementation
func (algo *Algo) Reset(conf Conf, data []Elemt) error {
	algo.conf = conf
	var impl, err = algo.impl.Reset(&conf, data)
	algo.impl = impl
	return err
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
		err = algo.impl.Push(elemt)
	}
	return
}

// ARun executes the algorithm and notify the user with a callback, timed by a time to callback (ttc) integer
func (algo *Algo) ARun(async bool, notifier Notifier) (err error) {
	if async {
		err = algo.impl.SetAsync()
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

// SetConf switches of configuration
func (algo *Algo) SetConf(conf Conf) (err error) {
	if algo.status == Running {
		err = errors.New("Impossible to switch configuration while running")
	} else {
		algo.conf = conf
	}
	return
}

// SetSpace switches of space
func (algo *Algo) SetSpace(space Space) (err error) {
	if algo.status == Running {
		err = errors.New("Impossible to switch configuration while running")
	} else {
		algo.space = space
	}
	return
}

// Conf returns configuration
func (algo Algo) Conf() Conf {
	return algo.conf
}

// Impl returns impl
func (algo Algo) Impl() Impl {
	return algo.impl
}

// Space returns space
func (algo Algo) Space() Space {
	return algo.space
}

// Predict the cluster for a new observation
func (algo *Algo) Predict(elemt Elemt, push bool) (pred Elemt, label int, err error) {
	var clust Clust
	clust, err = algo.Centroids()

	if err == nil {
		pred, label, _ = clust.Assign(elemt, algo.space)
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
	algo.centroids, err = algo.impl.Init(algo.conf, algo.space)
	if err == nil {
		var updater = algo.updateCentroids(notifier)
		err = algo.impl.Run(
			algo.conf,
			algo.space,
			algo.centroids,
			updater,
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
