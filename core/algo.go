// Package core proposes a generic framework that executes online clustering algorithm.
package core

import (
	"distclus/figures"
	"log"
	"sync"
	"sync/atomic"
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
	Predict(elemt Elemt) (Elemt, int, error)
	Run(bool) error
	Pause() error
	Play() error
	Close() error
	RuntimeFigures() (figures.RuntimeFigures, error)
}

// StatusNotifier for being notified by Online clustering change status
type StatusNotifier = func(ClustStatus, error)

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	conf           ImplConf
	impl           Impl
	space          Space
	centroids      Clust
	status         ClustStatus
	statusChannel  chan ClustStatus
	mutex          sync.RWMutex
	runtimeFigures figures.RuntimeFigures
	statusNotifier StatusNotifier
	async          bool
	newData        uint64
}

// AlgoConf algorithm configuration
type AlgoConf interface{}

// NewAlgo creates a new algorithm instance
func NewAlgo(conf ImplConf, impl Impl, space Space) *Algo {
	conf.AlgoConf().Verify()
	return &Algo{
		conf:           conf,
		impl:           impl,
		space:          space,
		status:         Created,
		statusChannel:  make(chan ClustStatus),
		statusNotifier: conf.AlgoConf().StatusNotifier,
	}
}

func (algo *Algo) setStatus(status ClustStatus, err error) {
	atomic.StoreInt64(&algo.status, status)
	if algo.statusNotifier != nil {
		go algo.statusNotifier(status, err)
	}
}

// Centroids Get the centroids currently found by the algorithm
func (algo *Algo) Centroids() (centroids Clust, err error) {
	switch atomic.LoadInt64(&algo.status) {
	case Created:
		err = ErrNotStarted
	default:
		algo.mutex.RLock()
		defer algo.mutex.RUnlock()
		centroids = algo.centroids
	}
	return
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	switch atomic.LoadInt64(&algo.status) {
	case Closed:
		err = ErrEnded
	default:
		err = algo.impl.Push(elemt)
		if err == nil {
			if algo.async {
				atomic.AddUint64(&algo.newData, 1)
			}
			err = algo.wakeUp()
		}
	}
	return
}

// Run executes the algorithm in batch mode
func (algo *Algo) Run(async bool) (err error) {
	algo.async = async
	err = algo.tryInit()
	if err == nil {
		err = algo.runIfReady()
	}
	return
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
	var status = algo.Status()
	if status == Running || status == Idle || status == Sleeping {
		algo.statusChannel <- Closed
		algo.setStatus(Closed, nil)
	} else {
		err = ErrNotRunning
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	var status = atomic.LoadInt64(&algo.status)
	if status == Running || status == Ready {
		algo.statusChannel <- Idle
		algo.setStatus(Idle, nil)
	} else {
		err = ErrNotRunning
	}
	return
}

// Play algorithm execution after idle status
func (algo *Algo) Play() (err error) {
	if atomic.LoadInt64(&algo.status) == Idle {
		algo.statusChannel <- Running
		algo.setStatus(Running, nil)
	} else {
		err = ErrNotIdle
	}
	return
}

func (algo *Algo) wakeUp() (err error) {
	if atomic.LoadInt64(&algo.status) == Sleeping {
		algo.statusChannel <- Running
		algo.setStatus(Running, nil)
	}
	return
}

// RuntimeFigures returns specific algo properties
func (algo *Algo) RuntimeFigures() (figures figures.RuntimeFigures, err error) {
	switch atomic.LoadInt64(&algo.status) {
	case Created:
		err = ErrNotRunning
	default:
		algo.mutex.RLock()
		defer algo.mutex.RUnlock()
		figures = algo.runtimeFigures
	}
	return
}

// Conf returns configuration
func (algo *Algo) Conf() ImplConf {
	return algo.conf
}

// Impl returns impl
func (algo *Algo) Impl() Impl {
	return algo.impl
}

// Status returns the status of the algorithm
func (algo *Algo) Status() ClustStatus {
	return atomic.LoadInt64(&algo.status)
}

func (algo *Algo) tryInit() (err error) {
	if atomic.LoadInt64(&algo.status) == Created {
		if algo.async {
			err = algo.impl.SetOC()
		}
		algo.centroids, err = algo.impl.Init(algo.conf, algo.space, algo.centroids)
		if err == nil {
			algo.setStatus(Ready, nil)
		}
	}
	return
}

func (algo *Algo) runIfReady() (err error) {
	if atomic.LoadInt64(&algo.status) == Ready {
		err = algo.run()
	} else {
		err = ErrNotReady
	}
	return
}

func (algo *Algo) run() (err error) {
	if algo.async {
		go algo.runAsync()
	} else {
		err = algo.runSync()
	}
	return
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) runSync() (err error) {
	algo.setStatus(Running, nil)

	var conf = algo.conf.AlgoConf()
	if conf.Iter == 0 && !algo.async {
		err = ErrInfiniteIterations
	}

	if err == nil {
		var centroids Clust
		var runtimeFigures figures.RuntimeFigures
		var start = time.Now()
		var iterFreq time.Duration
		if conf.IterFreq > 0 {
			iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
		}
		var lastIterationTime time.Time
		var newData = atomic.LoadUint64(&algo.newData)

		var status = Running

		for iterations := 0; status == Running && err == nil; iterations++ {
			select { // check for algo status update
			case status = <-algo.statusChannel:
				switch status {
				case Idle:
					status = <-algo.statusChannel
				case Closed:
				}
			default: // try to run the implementation
				if conf.Timeout > 0 && time.Now().Sub(start).Seconds() > conf.Timeout { // check timeout
					algo.setStatus(Closed, nil)
					err = ErrTimeOut
				} else { // run implementation
					lastIterationTime = time.Now()
					centroids, runtimeFigures, err = algo.impl.Iterate(
						algo.conf,
						algo.space,
						algo.centroids,
					)
					if centroids == nil { // impl has finished
						if !algo.async {
							status = Closed
							algo.setStatus(Closed, err)
						}
					} else if err == nil { // save run results
						algo.saveIterContext(centroids, runtimeFigures, iterations)
					}
					if (!algo.async) && conf.Iter > 0 && conf.Iter < iterations {
						algo.setStatus(Ready, nil)
						status = Ready
					}
				}
				if err == nil && atomic.LoadInt64(&algo.status) == Running {
					// temporize iteration
					if conf.IterFreq > 0 { // with iteration freqency
						var diff = iterFreq - time.Now().Sub(lastIterationTime)
						if diff > 0 {
							algo.setStatus(Idle, nil)
							time.Sleep(diff)
							algo.setStatus(Running, nil)
						}
					}
					if algo.async { // check online clustering activity
						var algoNewData = atomic.LoadUint64(&algo.newData)
						if newData < algoNewData {
							iterations = 0
							newData = algoNewData
						} else if newData == algoNewData {
							if (conf.Iter > 0 && iterations > conf.Iter) || (conf.DataPerIter > 0 && algoNewData < uint64(conf.DataPerIter)) {
								algo.setStatus(Sleeping, nil)
								status = <-algo.statusChannel
								atomic.StoreUint64(&algo.newData, 0)
								newData = 0
							}
						}
					}
				}
			}
		}
	}

	if err != nil {
		algo.setStatus(Failed, err)
	}
	if algo.async { // empty stack if necessary
		select {
		case <-algo.statusChannel:
			algo.setStatus(Closed, nil)
		default:
		}
	}
	return
}

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *Algo) runAsync() {
	var err error
	for atomic.LoadInt64(&algo.status) == Ready {
		err = algo.runSync()
		if err != nil {
			log.Println(err)
		}
	}
}

func (algo *Algo) saveIterContext(centroids Clust, runtimeFigures figures.RuntimeFigures, iterations int) {
	if atomic.LoadInt64(&algo.status) == Running {
		if runtimeFigures != nil {
			runtimeFigures[figures.Iterations] = figures.Value(iterations)
		} else {
			runtimeFigures = figures.RuntimeFigures{
				figures.Iterations: figures.Value(iterations),
			}
		}
		algo.mutex.Lock()
		algo.centroids = centroids
		algo.runtimeFigures = runtimeFigures
		algo.mutex.Unlock()
	}
}
