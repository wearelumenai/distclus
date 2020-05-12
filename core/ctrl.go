package core

import (
	"fmt"
	"math"
	"time"

	"github.com/wearelumenai/distclus/figures"
)

// OCCtrl online clustring controller
type OCCtrl interface {
	Init() error                               // initialize algo centroids with impl strategy
	Play(Finishing, time.Duration) error       // play (with x iterations if given, otherwise depends on conf.Iter/conf.IterPerData, and maximal duration in ns if given, otherwise conf.Timeout) the algorithm
	Pause() error                              // pause the algorithm (idle)
	Wait(Finishing, time.Duration) error       // wait (max x iterations if given and maximal duration in ns if given) for algorithm sleeping, ready or failed
	Stop() error                               // stop the algorithm
	Push(Elemt) error                          // add element
	Predict(elemt Elemt) (Elemt, int, float64) // input elemt centroid/label with distance to closest centroid
	Batch(Finishing, time.Duration) error      // execute (x iterations if given, otherwise depends on conf.Iter/conf.IterPerData) in batch mode (do play, wait, then stop)
	Reconfigure(Conf, Space) error             // reconfigure the online clust
	Copy(Conf, Space) (OnlineClust, error)     // make a copy of this algo with new configuration and space
	Alive() bool                               // true if algorithm is alive
	SetStatusNotifier(StatusNotifier)
	IsFinished(Finishing) bool
}

// Alive True if
func (algo *Algo) Alive() bool {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.status.Status >= Running && algo.status.Error == nil
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()

	err = algo.impl.Push(elemt, algo)
	if err == nil {
		algo.newData++
		algo.pushedData++
		algo.lastDataTime = time.Now().Unix()
		// try to play if waiting
		var conf = algo.conf.Ctrl()
		if algo.isStatus(Finished) && conf.DataPerIter > 0 && conf.DataPerIter <= algo.newData {
			algo.Play(nil, 0)
		}
	}

	return
}

func (algo *Algo) isStatus(statuses ...ClustStatus) (is bool) {
	algo.model.RLock()
	defer algo.model.RUnlock()
	var algoStatus = algo.status
	for _, status := range statuses {
		if status == algoStatus.Status {
			is = true
			break
		}
	}
	return
}

func (algo *Algo) playing() bool {
	return algo.isStatus(Running, Idle)
}

// Batch executes the algorithm in batch mode
func (algo *Algo) Batch(finishing Finishing, timeout time.Duration) (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	if algo.conf.Ctrl().Iter == 0 && finishing == nil && algo.finishing == nil {
		err = ErrNeverConverge
	} else {
		if !algo.playing() {
			algo.succeedOnce = false
			err = algo.play(finishing, timeout)
			if err == nil {
				err = algo.Wait(finishing, 0)
				if err == nil {
					if algo.timeout != nil {
						algo.timeout.Disable()
					}
				}
			}
		} else {
			err = ErrRunning
		}
	}
	return
}

// Init initialize centroids and set status to Ready
func (algo *Algo) Init() (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	err = algo.init(true)
	return
}

func (algo *Algo) init(lock bool) (err error) {
	if lock {
		algo.model.RLock()
		defer algo.model.RUnlock()
	}
	if algo.status.Status == Created || algo.status.Error != nil {
		algo.setStatus(OCStatus{Status: Initializing})
		algo.model.Lock()
		algo.centroids, err = algo.impl.Init(algo)
		algo.model.Unlock()
		if err == nil {
			algo.setStatus(OCStatus{Status: Ready, Error: err})
		} else {
			algo.setStatus(OCStatus{Status: Finished, Error: err})
		}
	} else {
		err = ErrAlreadyCreated
	}
	return
}

// Play the algorithm in online mode
func (algo *Algo) Play(finishing Finishing, timeout time.Duration) (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	return algo.play(finishing, timeout)
}

func (algo *Algo) play(finishing Finishing, timeout time.Duration) (err error) {
	algo.model.RLock()
	defer algo.model.RUnlock()
	switch algo.status.Status {
	case Idle:
		algo.sendStatus(OCStatus{Status: Running})
	case Finished:
		fallthrough
	case Created:
		if algo.status.Status == Created || algo.status.Error != nil {
			err = algo.init(false)
			if err != nil {
				return
			}
		}
		fallthrough
	case Ready:
		if !algo.IsFinished(finishing) {
			go algo.run(finishing)
			algo.sendStatus(OCStatus{Status: Running})
			if algo.timeout != nil {
				algo.timeout.Disable()
			}
			var interruptionTimeout time.Duration
			if timeout > 0 {
				interruptionTimeout = timeout
			} else if algo.Conf().Ctrl().Timeout > 0 { // && !algo.succeedOnce {
				interruptionTimeout = algo.Conf().Ctrl().Timeout
			}
			if interruptionTimeout > 0 {
				algo.timeout = InterruptionTimeout(interruptionTimeout, algo.interrupt)
			}
		}
	case Running:
		err = ErrRunning
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	if algo.Status().Status != Running || !algo.sendStatus(OCStatus{Status: Idle}) {
		err = ErrNotRunning
	}

	return
}

func (algo *Algo) canNeverConverge() bool {
	var conf = algo.conf.Ctrl()
	return algo.timeout == nil && algo.iterToRun == 0 && ((conf.Iter == 0 && !algo.succeedOnce) ||
		(algo.succeedOnce && conf.IterPerData == 0)) && conf.DataPerIter == 0 && algo.finishing == nil
}

// Wait for online ending status
func (algo *Algo) Wait(finishing Finishing, timeout time.Duration) (err error) {
	var status = algo.Status()
	err = status.Error
	if err == nil {
		if status.Status == Running {
			if algo.canNeverConverge() {
				return ErrNeverConverge
			}
			err = WaitTimeout(finishing, timeout, algo, algo.ackChannel)
		} else {
			err = ErrNotRunning
		}
	}
	return
}

// interrupt the algorithm
func (algo *Algo) interrupt(status OCStatus) (err error) {
	switch algo.status.Status {
	case Initializing:
		fallthrough
	case Idle:
		fallthrough
	case Running:
		algo.sendStatus(status)
		<-algo.ackChannel
		err = algo.status.Error
	default:
		err = ErrNotRunning
	}
	return
}

// Stop the algorithm
func (algo *Algo) Stop() (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	return algo.interrupt(OCStatus{Status: Finished})
}

// Predict the cluster for a new observation
func (algo *Algo) Predict(elemt Elemt) (pred Elemt, label int, dist float64) {
	var clust = algo.Centroids()
	pred, label, dist = clust.Assign(elemt, algo.space)
	return
}

func (algo *Algo) initIterToRun(iter int) {
	algo.model.Lock()
	defer algo.model.Unlock()
	if iter > 0 {
		algo.iterToRun = iter
	} else if algo.succeedOnce {
		algo.iterToRun = algo.conf.Ctrl().IterPerData
	} else {
		algo.iterToRun = algo.conf.Ctrl().Iter
	}
}

func (algo *Algo) recover(newData int, start time.Time) {
	var recovery = recover()
	if recovery != nil {
		var err = fmt.Errorf("%v", recovery)
		algo.setStatus(OCStatus{Status: Finished, Error: err})
	}
	algo.model.Lock()
	algo.succeedOnce = algo.status.Error == nil
	algo.newData = int(math.Max(0, float64(algo.newData-newData)))
	algo.duration += time.Now().Sub(start)
	algo.iterToRun = 0
	algo.model.Unlock()

	select { // free user send status
	case <-algo.statusChannel:
	default:
	}
	select { // close ack channel
	case <-algo.ackChannel:
	default:
		close(algo.ackChannel)
	}
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) run(finishing Finishing) {
	algo.ackChannel = make(chan bool)

	var err error
	var conf = algo.conf.Ctrl()
	var centroids Clust
	var runtimeFigures figures.RuntimeFigures
	var iterFreq time.Duration
	if conf.IterFreq > 0 {
		iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
	}
	var lastIterationTime = time.Now()

	var iterations = 0

	var newData int

	algo.newData = 0

	var start = time.Now()
	var duration time.Duration

	defer algo.recover(newData, start)

	algo.receiveStatus()

	for algo.status.Status == Running && !algo.IsFinished(finishing) {
		select { // check for algo status update
		case status := <-algo.statusChannel:
			algo.setStatus(status)
			if status.Status == Idle {
				algo.ackChannel <- true
				algo.receiveStatus()
			}
		default:
			// run implementation
			newData = algo.newData
			centroids, runtimeFigures, err = algo.impl.Iterate(algo)
			duration = time.Now().Sub(start)
			if err == nil {
				if centroids != nil { // an iteration has been executed
					algo.totalIterations++
					iterations++
					algo.saveIterContext(
						centroids, runtimeFigures,
						iterations,
						duration,
					)
				}
				// temporize iteration
				if iterFreq > 0 { // with iteration freqency
					var diff = iterFreq - time.Now().Sub(lastIterationTime)
					time.Sleep(diff)
					lastIterationTime = time.Now()
				}
			} else { // impl has finished
				algo.setStatus(OCStatus{Status: Finished, Error: err})
			}
		}
	}
}

// IsFinished true iif input is finished with algo ctxt
func (algo *Algo) IsFinished(finishing Finishing) bool {
	return NewAnd(algo.finishing, finishing).IsFinished(algo)
}

func (algo *Algo) saveIterContext(centroids Clust, runtimeFigures figures.RuntimeFigures, iterations int, duration time.Duration) {
	if runtimeFigures == nil {
		runtimeFigures = figures.RuntimeFigures{}
	}
	runtimeFigures[figures.Iterations] = float64(algo.totalIterations)
	runtimeFigures[figures.LastIterations] = float64(iterations)
	runtimeFigures[figures.PushedData] = float64(algo.pushedData)
	runtimeFigures[figures.LastDuration] = float64(duration)
	runtimeFigures[figures.Duration] = float64(algo.duration + duration)
	runtimeFigures[figures.LastDataTime] = float64(algo.lastDataTime)
	algo.model.Lock()
	algo.centroids = centroids
	algo.runtimeFigures = runtimeFigures
	algo.model.Unlock()
}

// Reconfigure algo configuration and space
func (algo *Algo) Reconfigure(conf Conf, space Space) (err error) {
	algo.ctrl.RLock()
	defer algo.ctrl.RUnlock()
	impl, err := algo.impl.Copy(algo)
	if err == nil {
		algo.model.Lock()
		algo.impl = impl
		algo.conf = conf
		algo.space = space
		algo.model.Unlock()
	} else {
		algo.setStatus(OCStatus{Status: Finished, Error: err})
	}
	return
}

// Copy make a copy of this algo with new conf and space
func (algo *Algo) Copy(conf Conf, space Space) (oc OnlineClust, err error) {
	impl, err := algo.impl.Copy(algo)
	if err == nil {
		oc = NewAlgo(conf, impl, space)
	}
	return
}

// SetStatusNotifier change of statusNotifier
func (algo *Algo) SetStatusNotifier(statusNotifier StatusNotifier) {
	algo.model.Lock()
	defer algo.model.Unlock()
	algo.statusNotifier = statusNotifier
}
