// Package core proposes a generic framework that executes online clustering algorithm.
package core

import (
	"distclus/figures"
	"log"
	"math"
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
	Init() error  // initialize algo centroids with impl strategy
	Play() error  // play the algorithm
	Pause() error // pause the algorithm (idle)
	Wait() error  // wait for algorithm sleeping, ready or failed
	Stop() error  // stop the algorithm
	Close() error
	Push(elemt Elemt) error // add element
	// only if algo has runned once
	Centroids() (Clust, error) // clustering result
	Conf() ImplConf
	Impl() Impl
	Predict(elemt Elemt) (Elemt, int, error)         // input elemt centroid/label
	Batch() error                                    // execute in batch mode (do play, wait, then stop)
	Running() bool                                   // true iif algo is running (running, idle and sleeping)
	Status() ClustStatus                             // algo status
	FailedError() error                              // error in case of failure
	RuntimeFigures() (figures.RuntimeFigures, error) // clustering figures
	Reconfigure(ImplConf, Space) error               // reconfigure the online clust
	Copy(ImplConf, Space) (OnlineClust, error)       // make a copy of this algo with new configuration and space
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
	newData        int64
	pushedData     int64
	failedError    error
	iterations     int
	duration       time.Duration
	lastDataTime   int64
	succeedOnce    bool
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
func (algo *Algo) sendStatus(status ClustStatus) (ok bool) {
	algo.statusChannel <- status
	_, ok = <-algo.ackChannel
	return
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
	return algo.status == Running || algo.status == Idle || algo.status == Sleeping || algo.status == Waiting
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	if algo.Status() == Closed {
		err = ErrClosed
	} else {
		err = algo.impl.Push(elemt, algo.Running())
		if err == nil {
			atomic.AddInt64(&algo.newData, 1)
			atomic.AddInt64(&algo.pushedData, 1)
			atomic.StoreInt64(&algo.lastDataTime, time.Now().Unix())
			// try to play
			if (!algo.Running()) && algo.conf.AlgoConf().DataPerIter > 0 && algo.conf.AlgoConf().DataPerIter <= int(atomic.LoadInt64(&algo.newData)) {
				algo.Play()
			}
		}
	}
	return
}

// Batch executes the algorithm in batch mode
func (algo *Algo) Batch() (err error) {
	if algo.conf.AlgoConf().Iter == 0 {
		err = ErrInfiniteIterations
	} else {
		switch algo.Status() {
		case Succeed:
			fallthrough
		case Failed:
			fallthrough
		case Waiting:
			algo.succeedOnce = false
			fallthrough
		case Created:
			fallthrough
		case Ready:
			err = algo.Play()
			if err == nil {
				err = algo.Wait()
				if err == nil {
					algo.setStatus(Succeed, nil)
				}
			}
		case Closed:
			err = ErrClosed
		default:
			err = ErrRunning
		}
	}
	return
}

// Init initialize centroids and set status to Ready
func (algo *Algo) Init() (err error) {
	switch algo.Status() {
	case Created:
		algo.setStatus(Initializing, nil)
		algo.centroids, err = algo.impl.Init(algo.conf, algo.space, algo.centroids)
		if err == nil {
			algo.setStatus(Ready, nil)
		} else {
			algo.setStatus(Created, err)
		}
	case Closed:
		err = ErrClosed
	default:
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
	case Waiting:
		fallthrough
	case Succeed:
		if algo.Status() == Ready || algo.canIterate(0) {
			go algo.run()
			algo.sendStatus(Running)
		} else {
			err = ErrNotIterate
		}
	case Idle:
		algo.sendStatus(Running)
	case Sleeping:
		err = ErrSleeping
	case Running:
		err = ErrRunning
	case Closed:
		err = ErrClosed
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	switch algo.Status() {
	case Sleeping:
		fallthrough
	case Running:
		if !algo.sendStatus(Idle) {
			err = ErrNotRunning
		}
	case Waiting:
		err = ErrWaiting
	case Idle:
		err = ErrIdle
	case Closed:
		err = ErrClosed
	default:
		err = ErrNotRunning
	}
	return
}

func (algo *Algo) canNeverEnd() bool {
	var conf = algo.conf.AlgoConf()
	return ((conf.Iter == 0 && !algo.succeedOnce) ||
		(algo.succeedOnce && conf.IterPerData == 0)) && conf.DataPerIter == 0
}

// Wait for online ending status
func (algo *Algo) Wait() (err error) {
	switch algo.Status() {
	case Idle:
		err = ErrIdle
	case Sleeping:
		fallthrough
	case Running:
		if algo.canNeverEnd() {
			return ErrNeverEnd
		}
		<-algo.ackChannel
		fallthrough
	case Failed:
		err = algo.failedError
	case Succeed:
	case Waiting:
	case Created:
		fallthrough
	case Ready:
		err = ErrNotStarted
	case Closed:
		err = ErrClosed
	}
	return
}

// Stop the algorithm
func (algo *Algo) Stop() (err error) {
	switch algo.Status() {
	case Idle:
		fallthrough
	case Running:
		fallthrough
	case Sleeping:
		algo.sendStatus(Stopping)
		<-algo.ackChannel
		err = algo.failedError
	case Created:
		fallthrough
	case Ready:
		err = ErrNotStarted
	case Closed:
		err = ErrClosed
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
	algo.ackChannel = make(chan bool)

	algo.receiveStatus()

	var err error
	var conf = algo.conf.AlgoConf()
	var centroids Clust
	var runtimeFigures figures.RuntimeFigures
	var iterFreq time.Duration
	if conf.IterFreq > 0 {
		iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
	}
	var lastIterationTime time.Time

	var iterations = 0

	var newData int64

	atomic.StoreInt64(&algo.newData, 0)
	var start = time.Now()
	var duration time.Duration

	for algo.status == Running && algo.canIterate(iterations) {
		select { // check for algo status update
		case status := <-algo.statusChannel:
			algo.setStatus(status, nil)
			if status == Idle || status == Reconfiguring {
				algo.ackChannel <- true
				algo.receiveStatus()
			}
		default:
			if conf.Timeout > 0 && time.Duration(conf.Timeout) <= (duration+algo.duration) { // check timeout
				algo.setStatus(Failed, ErrTimeOut)
			} else {
				// run implementation
				lastIterationTime = time.Now()
				newData = atomic.LoadInt64(&algo.newData)
				centroids, runtimeFigures, err = algo.impl.Iterate(
					algo.conf,
					algo.space,
					algo.centroids,
				)
				if err == nil {
					algo.iterations++
					iterations++
					duration = time.Now().Sub(start)
					algo.saveIterContext(
						centroids, runtimeFigures,
						iterations,
						duration,
					)
					// temporize iteration
					if conf.IterFreq > 0 { // with iteration freqency
						var diff = iterFreq - time.Now().Sub(lastIterationTime)
						if diff > 0 {
							algo.setStatus(Sleeping, nil)
							time.Sleep(diff)
							algo.setStatus(Running, nil)
						}
					}
				} else { // impl has finished
					algo.setStatus(Failed, err)
				}
			}
		}
	}

	atomic.StoreInt64(
		&algo.newData,
		int64(math.Max(0, float64(atomic.LoadInt64(&algo.newData)-newData))),
	)
	algo.duration += time.Now().Sub(start)
	algo.succeedOnce = true

	if algo.status == Failed {
		log.Println(algo.failedError)
	} else {
		algo.setStatus(Waiting, nil)
	}

	close(algo.ackChannel)

	select { // free user send status
	case <-algo.statusChannel:
	default:
	}
}

// FailedError is the error in case of algo failure
func (algo *Algo) FailedError() (err error) {
	algo.mutex.RLock()
	defer algo.mutex.RUnlock()
	return algo.failedError
}

func (algo *Algo) canIterate(iterations int) bool {
	var conf = algo.conf.AlgoConf()
	var iter = conf.Iter
	if algo.succeedOnce {
		iter = conf.IterPerData
	}
	var iterDone = iter == 0 || iterations < iter
	var dataPerIterDone = conf.DataPerIter == 0 || (int64(conf.DataPerIter) <= atomic.LoadInt64(&algo.newData))
	return iterDone && dataPerIterDone
}

func (algo *Algo) saveIterContext(centroids Clust, runtimeFigures figures.RuntimeFigures, iterations int, duration time.Duration) {
	if runtimeFigures == nil {
		runtimeFigures = figures.RuntimeFigures{}
	}
	runtimeFigures[figures.Iterations] = float64(algo.iterations)
	runtimeFigures[figures.LastIterations] = float64(iterations)
	runtimeFigures[figures.PushedData] = float64(algo.pushedData)
	runtimeFigures[figures.LastDuration] = float64(duration)
	runtimeFigures[figures.Duration] = float64(algo.duration + duration)
	runtimeFigures[figures.LastDataTime] = float64(atomic.LoadInt64(&algo.lastDataTime))
	algo.mutex.Lock()
	algo.centroids = centroids
	algo.runtimeFigures = runtimeFigures
	algo.mutex.Unlock()
}

func (algo *Algo) reconfigure(conf ImplConf, space Space) (err error) {
	impl, err := algo.impl.Copy(conf, space)
	if err == nil {
		algo.impl = impl
		algo.conf = conf
		algo.space = space
	}
	return
}

// Reconfigure algo configuration and space
func (algo *Algo) Reconfigure(conf ImplConf, space Space) (err error) {
	var status = algo.Status()
	switch status {
	case Created:
		fallthrough
	case Ready:
		err = ErrNotStarted
	case Failed:
		fallthrough
	case Succeed:
		fallthrough
	case Waiting:
		fallthrough
	case Idle:
		algo.setStatus(Reconfiguring, nil)
		err = algo.reconfigure(conf, space)
		var newStatus = algo.Status()
		if newStatus == Reconfiguring {
			newStatus = status
		}
		algo.setStatus(newStatus, nil)
	case Running:
		fallthrough
	case Stopping:
		fallthrough
	case Sleeping:
		var sent = algo.sendStatus(Reconfiguring)
		err = algo.reconfigure(conf, space)
		if sent {
			var newStatus = algo.Status()
			if newStatus == Reconfiguring {
				newStatus = status
			}
			algo.sendStatus(newStatus)
		}
	case Closed:
		err = ErrClosed
	}
	return
}

// Copy make a copy of this algo with new conf and space
func (algo *Algo) Copy(conf ImplConf, space Space) (oc OnlineClust, err error) {
	impl, err := algo.impl.Copy(conf, space)
	if err == nil {
		oc = NewAlgo(conf, impl, space)
	}
	return
}

// Close close the algorithm
func (algo *Algo) Close() (err error) {
	algo.Stop()
	algo.setStatus(Closed, nil)
	return
}
