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
// an element can't be pushed if the algorithm is stopped,
// a prediction can't be done before the algorithm is run,
// no centroid can be returned before the algorithm is run.
type OnlineClust interface {
	Play() error            // play the algorithm
	Pause() error           // pause the algorithm (idle)
	Wait() error            // wait for algorithm sleeping, ready or failed
	Stop() error            // stop the algorithm
	Push(elemt Elemt) error // add element
	// only if algo has runned once
	Centroids() (Clust, error)                       // clustering result
	Predict(elemt Elemt) (Elemt, int, error)         // input elemt centroid/label
	Batch() error                                    // execute in batch mode (do play, wait, then stop)
	Running() bool                                   // true iif algo is running (running, idle and sleeping)
	Status() ClustStatus                             // algo status
	RuntimeFigures() (figures.RuntimeFigures, error) // clustering figures
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
	waitNotifier   chan error
	newData        int64
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
		waitNotifier:   make(chan error),
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

// Running is true only if the algorithm is running
func (algo *Algo) Running() bool {
	var status = algo.Status()
	return status == Running || status == Idle || status == Sleeping
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	err = algo.impl.Push(elemt, algo.Running())
	if err == nil {
		atomic.AddInt64(&algo.newData, 1)
		err = algo.wakeUp()
	}
	return
}

// Batch executes the algorithm in batch mode
func (algo *Algo) Batch() (err error) {
	if algo.conf.AlgoConf().Iter == 0 {
		err = ErrInfiniteIterations
	} else if algo.Running() {
		err = ErrRunning
	} else {
		err = algo.Play()
		if err == nil {
			err = algo.Wait()
			if err == nil {
				err = algo.Stop()
			}
		}
	}
	return
}

// Play the algorithm in online mode
func (algo *Algo) Play() (err error) {
	switch atomic.LoadInt64(&algo.status) {
	case Created:
		algo.centroids, err = algo.impl.Init(algo.conf, algo.space, algo.centroids)
		if err == nil {
			algo.setStatus(Ready, nil)
		} else {
			return
		}
		fallthrough
	case Failed:
		fallthrough
	case Ready:
		algo.statusChannel = make(chan ClustStatus)
		algo.waitNotifier = make(chan error)
		go algo.run()
		<-algo.statusChannel // wait run ack
	case Idle:
		algo.statusChannel <- Running
		algo.setStatus(Running, nil)
	case Sleeping:
		err = ErrSleeping
	case Running:
		err = ErrRunning
	case Stopping:
		err = ErrStopping
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

// Stop Stops the algorithm
func (algo *Algo) Stop() (err error) {
	if algo.Running() {
		algo.setStatus(Stopping, nil)
		algo.statusChannel <- Stopping
		<-algo.statusChannel
	} else {
		err = ErrNotRunning
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	var status = atomic.LoadInt64(&algo.status)
	if status == Running || status == Sleeping {
		algo.statusChannel <- Idle
		algo.setStatus(Idle, nil)
	} else {
		err = ErrNotRunning
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

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) run() (err error) {
	algo.setStatus(Running, nil)

	var conf = algo.conf.AlgoConf()

	defer (func() { // clean algorithm
		atomic.StoreInt64(&algo.newData, 0)
		select { // unlock wait notifier if blocked
		case algo.waitNotifier <- err:
		default:
		}
		close(algo.statusChannel)
		close(algo.waitNotifier)
		algo.Stop()
	})()

	algo.statusChannel <- Running // unlock runIfReady pause

	var centroids Clust
	var runtimeFigures figures.RuntimeFigures
	var start = time.Now()
	var iterFreq time.Duration
	if conf.IterFreq > 0 {
		iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
	}
	var lastIterationTime time.Time
	var newData = atomic.LoadInt64(&algo.newData)

	var status = Running
	var iterations = 0
	var iterationsPerSleep = 0

	for (status == Running || status == Idle) && err == nil {
		select { // check for algo status update
		case status = <-algo.statusChannel:
			switch status {
			case Idle:
				status = <-algo.statusChannel
			case Stopping:
			}
		default: // try to run the implementation
			if conf.Timeout > 0 && time.Now().Sub(start).Seconds() > conf.Timeout { // check timeout
				algo.setStatus(Stopping, nil)
				err = ErrTimeOut
			} else { // run implementation
				lastIterationTime = time.Now()
				centroids, runtimeFigures, err = algo.impl.Iterate(
					algo.conf,
					algo.space,
					algo.centroids,
				)
				if err == nil {
					if centroids != nil { // impl has finished
						iterationsPerSleep++
						iterations++
						algo.saveIterContext(centroids, runtimeFigures, iterations)
						// temporize iteration
						if conf.IterFreq > 0 { // with iteration freqency
							var diff = iterFreq - time.Now().Sub(lastIterationTime)
							if diff > 0 {
								time.Sleep(diff)
							}
						}
						var algoNewData = atomic.LoadInt64(&algo.newData)
						if newData < algoNewData {
							newData = algoNewData
							iterationsPerSleep = 0
						} else if newData == algoNewData {
							if (conf.Iter > 0 && iterationsPerSleep >= conf.Iter) || (conf.DataPerIter > 0 && int64(conf.DataPerIter) >= algoNewData) {
								algo.setStatus(Sleeping, nil)
								algo.notifyWaiters()
								status = <-algo.statusChannel
								atomic.StoreInt64(&algo.newData, 0)
								newData = 0
								iterationsPerSleep = 0
							}
						}
					} else { // impl has finished
						algo.setStatus(Stopping, nil)
						status = Stopping
					}
				}
			}
		}
	}

	if err != nil {
		algo.setStatus(Failed, err)
		log.Println(err)
	} else if status == Stopping {
		algo.setStatus(Ready, nil)
	}
	return
}

func (algo *Algo) notifyWaiters() {
	select {
	case algo.waitNotifier <- nil:
	default:
	}
}

// Wait for online sleeping/ending status
func (algo *Algo) Wait() (err error) {
	switch atomic.LoadInt64(&algo.status) {
	case Idle:
		err = ErrIdle
	case Running:
		if algo.conf.AlgoConf().Iter == 0 || algo.conf.AlgoConf().DataPerIter == 0 {
			err = ErrNeverSleeping
		} else {
			err = <-algo.waitNotifier
		}
	case Sleeping:
	default:
		err = ErrNotRunning
	}
	return
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
