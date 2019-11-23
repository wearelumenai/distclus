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
	Init() error            // initialize algo centroids with impl strategy
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
	ackChannel     chan bool
	mutex          sync.RWMutex
	runtimeFigures figures.RuntimeFigures
	statusNotifier StatusNotifier
	waitChannel    chan error
	newData        int64
	failedError    error
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
		ackChannel:     make(chan bool),
		statusNotifier: conf.AlgoConf().StatusNotifier,
	}
}

// change of status
func (algo *Algo) setStatus(status ClustStatus, err error) {
	algo.mutex.Lock()
	algo.failedError = err
	algo.status = status
	algo.mutex.Unlock()
	if algo.statusNotifier != nil {
		algo.statusNotifier(status, err)
	}
}

// receiveStatus status from main routine
func (algo *Algo) receiveStatus() {
	var status = <-algo.statusChannel
	algo.setStatus(status, nil)
	algo.ackChannel <- true
}

// sendStatus status to run go routine
func (algo *Algo) sendStatus(status ClustStatus) {
	algo.statusChannel <- status
	<-algo.ackChannel
}

// Centroids Get the centroids currently found by the algorithm
func (algo *Algo) Centroids() (centroids Clust, err error) {
	algo.mutex.RLock()
	defer algo.mutex.RUnlock()
	switch algo.Status() {
	case Created:
		err = ErrNotStarted
	default:
		centroids = algo.centroids
	}
	return
}

// Running is true only if the algorithm is running
func (algo *Algo) Running() bool {
	algo.mutex.RLock()
	defer algo.mutex.RUnlock()
	return algo.status == Running || algo.status == Idle || algo.status == Sleeping
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	err = algo.impl.Push(elemt, algo.Running())
	if err == nil {
		atomic.AddInt64(&algo.newData, 1)
		// wakeup sleeping
		if algo.Status() == Sleeping {
			algo.sendStatus(Running)
		}
	}
	return
}

// Batch executes the algorithm in batch mode
func (algo *Algo) Batch() (err error) {
	if algo.conf.AlgoConf().Iter == 0 {
		err = ErrInfiniteIterations
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

// Init initialize centroids and set status to Ready
func (algo *Algo) Init() (err error) {
	if algo.Status() == Created {
		algo.centroids, err = algo.impl.Init(algo.conf, algo.space, algo.centroids)
		if err == nil {
			algo.setStatus(Ready, nil)
		}
	} else {
		err = ErrAlreadyCreated
	}
	return
}

// Play the algorithm in online mode
func (algo *Algo) Play() (err error) {
	switch algo.Status() {
	case Created:
		err = algo.Init()
		if err != nil {
			return
		}
		fallthrough
	case Ready:
		fallthrough
	case Failed:
		fallthrough
	case Succeed:
		go algo.run()
		fallthrough
	case Idle:
		fallthrough
	case Sleeping:
		algo.sendStatus(Running)
	case Running:
		err = ErrRunning
	case Stopping:
		err = ErrStopping
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	switch algo.Status() {
	case Sleeping:
		fallthrough
	case Running:
		algo.sendStatus(Idle)
	case Idle:
		err = ErrIdle
	default:
		err = ErrNotRunning
	}
	return
}

// Wait for online sleeping/ending status
func (algo *Algo) Wait() (err error) {
	switch algo.Status() {
	case Idle:
		err = ErrIdle
	case Running:
		if algo.conf.AlgoConf().Iter == 0 && algo.conf.AlgoConf().DataPerIter == 0 {
			return ErrNeverSleeping
		}
		fallthrough
	case Stopping:
		err = <-algo.waitChannel
		if err == nil {
			err = algo.failedError
		}
	case Sleeping:
	case Succeed:
	case Failed:
		err = algo.failedError
	case Created:
		fallthrough
	case Ready:
		err = ErrNotStarted
	}
	return
}

// Stop the algorithm
func (algo *Algo) Stop() (err error) {
	if algo.Running() {
		algo.sendStatus(Stopping)
		err = algo.Wait()
	} else {
		err = ErrNotRunning
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

// RuntimeFigures returns specific algo properties
func (algo *Algo) RuntimeFigures() (figures figures.RuntimeFigures, err error) {
	switch algo.Status() {
	case Created:
		err = ErrNotStarted
	default:
		algo.mutex.RLock()
		figures = algo.runtimeFigures
		algo.mutex.RUnlock()
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
	algo.mutex.RLock()
	defer algo.mutex.RUnlock()
	return algo.status
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) run() {
	var err error

	var conf = algo.conf.AlgoConf()

	algo.waitChannel = make(chan error)
	algo.ackChannel = make(chan bool)

	algo.receiveStatus()

	var centroids Clust
	var runtimeFigures figures.RuntimeFigures
	var start = time.Now()
	var iterFreq time.Duration
	if conf.IterFreq > 0 {
		iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
	}
	var lastIterationTime time.Time

	var iterations = 0
	var iterationsPerSleep = 0

	atomic.StoreInt64(&algo.newData, 0)

	for (algo.status == Running || algo.status == Idle) && err == nil {
		if algo.status == Idle {
			algo.receiveStatus()
		}
		select { // check for algo status update
		case status := <-algo.statusChannel:
			switch status {
			case Idle:
				algo.setStatus(Idle, nil)
				algo.ackChannel <- true
			case Stopping:
				algo.setStatus(Stopping, nil)
			}
		default:
			select {
			case algo.ackChannel <- true:
			default:
			}
			if conf.Timeout > 0 && time.Now().Sub(start).Seconds() > conf.Timeout { // check timeout
				err = ErrTimeOut
				algo.setStatus(Stopping, err)
			} else { // check sleeping
				var iterSleep = conf.Iter == 0 || (conf.Iter > 0 && conf.Iter <= iterationsPerSleep)
				var dataPerIterSleep = conf.DataPerIter == 0 || (int64(conf.DataPerIter) <= atomic.LoadInt64(&algo.newData))
				if (conf.DataPerIter > 0 || conf.Iter > 0) && (iterSleep && dataPerIterSleep) {
					algo.setStatus(Sleeping, nil)
					select { // wait for waiters
					case algo.waitChannel <- nil:
					default:
					}
					algo.receiveStatus()
					atomic.StoreInt64(&algo.newData, 0)
					iterationsPerSleep = 0
				} else {
					// run implementation
					lastIterationTime = time.Now()
					centroids, runtimeFigures, err = algo.impl.Iterate(
						algo.conf,
						algo.space,
						algo.centroids,
					)
					if err == nil {
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
					} else { // impl has finished
						algo.setStatus(Stopping, err)
					}
				}
			}
		}
	}

	if err == nil {
		algo.setStatus(Succeed, nil)
	} else {
		algo.setStatus(Failed, err)
		log.Println(err)
	}
	// close(algo.statusChannel)
	close(algo.waitChannel)
	close(algo.ackChannel)
}

func (algo *Algo) saveIterContext(centroids Clust, runtimeFigures figures.RuntimeFigures, iterations int) {
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
