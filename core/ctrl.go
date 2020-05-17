package core

import (
	"fmt"
	"time"
)

// OCCtrl online clustring controller
type OCCtrl interface {
	Init() error                               // initialize algo centroids with impl strategy
	Play() error                               // play the algorithm
	Pause() error                              // pause the algorithm (idle)
	Wait(Finishing, time.Duration) error       // wait for finishing condition and maximal duration. By default, finishing is ready/idle/finished status, and duration is infinite
	Stop() error                               // stop the algorithm
	Push(Elemt) error                          // add element
	Predict(elemt Elemt) (Elemt, int, float64) // input elemt centroid/label with distance to closest centroid
	Batch() error                              // batch mode (stop, play, wait then stop)
	Copy(Conf, Space) (OnlineClust, error)     // make a copy of this algo with new configuration and space
}

// Push a new observation in the algorithm
func (algo *Algo) Push(elemt Elemt) (err error) {
	err = algo.impl.Push(elemt, algo)
	if err == nil {
		algo.modelMutex.Lock()
		algo.pushedData++
		algo.lastDataTime = time.Now().Unix()
		var conf = algo.conf.Ctrl()
		if algo.Status().Value == Ready && conf.DataPerIter > 0 && conf.DataPerIter <= algo.newData {
			algo.newData = 0
		} else {
			algo.newData++
		}
		algo.updateRuntimeFigures()
		algo.modelMutex.Unlock()
		// try to play if waiting
		algo.modelMutex.RLock()
		defer algo.modelMutex.RUnlock()
		if algo.newData == 0 {
			algo.Play()
		}
	}
	return
}

// Batch executes the algorithm in batch mode
func (algo *Algo) Batch() (err error) {
	algo.Stop()
	err = algo.Play()
	if err == nil {
		err = algo.Wait(nil, 0)
		if err == nil {
			algo.Stop()
		}
	}
	return
}

// Init initialize centroids and set status to Ready
func (algo *Algo) Init() error {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	algo.statusMutex.Lock()
	defer algo.statusMutex.Unlock()
	return algo.init()
}

func (algo *Algo) init() (err error) {
	switch algo.status.Value {
	case Finished:
		algo.modelMutex.Lock()
		algo.pushedData = 0
		algo.duration = 0
		algo.lastDataTime = 0
		algo.iterations = 0
		algo.runtimeFigures = RuntimeFigures{}
		algo.updateRuntimeFigures()
		algo.modelMutex.Unlock()
		fallthrough
	case Created:
		algo.setStatus(NewOCStatus(Initializing), false)
		var centroids Clust
		centroids, err = algo.impl.Init(algo)
		algo.modelMutex.Lock()
		algo.centroids = centroids
		algo.modelMutex.Unlock()
		if err == nil {
			algo.setStatus(NewOCStatus(Ready), false)
		} else {
			algo.setStatus(NewOCStatusError(err), false)
		}
	default:
		err = ErrAlreadyCreated
	}
	return
}

// Play the algorithm in online mode
func (algo *Algo) Play() (err error) {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	return algo.play()
}

func (algo *Algo) play() (err error) {
	algo.statusMutex.Lock()
	switch algo.status.Value {
	case Idle:
		algo.statusMutex.Unlock()
		algo.sendStatus(NewOCStatus(Running))
	case Finished:
		fallthrough
	case Created:
		err = algo.init()
		if err != nil && err != ErrAlreadyCreated {
			return
		}
		err = nil
		fallthrough
	case Ready:
		go algo.run()
		algo.statusMutex.Unlock()
		algo.sendStatus(NewOCStatus(Running))
		if algo.timeout != nil {
			algo.timeout.Disable()
		}
		var interruptionTimeout = algo.Conf().Ctrl().Timeout
		if interruptionTimeout > 0 {
			algo.timeout = InterruptionTimeout(interruptionTimeout, algo.interrupt)
		}
	case Running:
		algo.statusMutex.Unlock()
		err = ErrRunning
	}
	return
}

// Pause the algorithm and set status to idle
func (algo *Algo) Pause() (err error) {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	algo.statusMutex.Lock()
	if algo.status.Value == Running {
		algo.statusMutex.Unlock()
		if !algo.sendStatus(NewOCStatus(Idle)) {
			err = ErrNotRunning
		}
	} else {
		err = ErrNotRunning
		algo.statusMutex.Unlock()
	}
	return
}

// CanNeverFinish true if finishing is nil, timeout is negative, conf.Timeout, conf.Iter and conf.IterPerData equal 0
func (algo *Algo) CanNeverFinish(finishing Finishing, timeout time.Duration) bool {
	var conf = algo.Conf().Ctrl()
	return finishing == nil && timeout <= 0 && conf.Finishing == nil && conf.Timeout <= 0 && conf.Iter == 0 && conf.IterPerData == 0
}

// Wait for online finishing predicate
func (algo *Algo) Wait(finishing Finishing, timeout time.Duration) (err error) {
	if algo.Status().Value == Running {
		if algo.CanNeverFinish(finishing, timeout) {
			err = ErrNeverFinish
		} else {
			err = WaitTimeout(finishing, timeout, algo)
		}
	} else {
		err = ErrNotRunning
	}
	return
}

// interrupt the algorithm
func (algo *Algo) interrupt(interruption error) (err error) {
	algo.statusMutex.Lock()
	switch algo.status.Value {
	case Ready:
		algo.setStatus(NewOCStatusError(interruption), false)
		algo.statusMutex.Unlock()
	case Idle:
		fallthrough
	case Running:
		algo.statusMutex.Unlock()
		algo.sendStatus(NewOCStatusError(interruption))
		err = algo.status.Error
	default:
		algo.statusMutex.Unlock()
		err = ErrNotAlive
	}
	return
}

// Stop the algorithm
func (algo *Algo) Stop() (err error) {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	return algo.interrupt(nil)
}

// Predict the cluster for a new observation
func (algo *Algo) Predict(elemt Elemt) (pred Elemt, label int, dist float64) {
	var clust = algo.Centroids()
	pred, label, dist = clust.Assign(elemt, algo.space)
	return
}

func (algo *Algo) recover(start time.Time) {
	algo.statusMutex.Lock()
	defer algo.statusMutex.Unlock()
	var recovery = recover()
	if recovery != nil {
		var err = fmt.Errorf("%v", recovery)
		algo.setStatus(NewOCStatusError(err), false)
	}
	// wait a few between client time processing between releasing satusMutex and sendStatus
	time.Sleep(100 * time.Millisecond)
	// update algo runtime figures
	algo.modelMutex.Lock()
	// algo.newData = int(math.Max(0, float64(algo.newData-newData)))
	var duration = time.Now().Sub(start)
	algo.duration += duration
	algo.runtimeFigures[Duration] = float64(algo.duration)
	algo.modelMutex.Unlock()
	// clean up channels
	select {
	// get client status if client quiered to stop or timeout interruption
	case status := <-algo.statusChannel:
		if recovery == nil {
			algo.setStatus(status, false)
		}
	default:
	}
	// free client ack channel
	select { // close ack channel
	case <-algo.ackChannel:
	default:
		close(algo.ackChannel)
	}
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
func (algo *Algo) run() {
	algo.ackChannel = make(chan bool)

	var err error
	var conf = algo.conf.Ctrl()
	var centroids = algo.centroids
	var runtimeFigures RuntimeFigures
	var iterFreq time.Duration
	if conf.IterFreq > 0 {
		iterFreq = time.Duration(float64(time.Second) / conf.IterFreq)
	}
	var lastIterationTime = time.Now()

	// var newData int

	var start = time.Now()
	var duration time.Duration

	var finishing Finishing = NewIterFinishing(conf.Iter, conf.IterPerData)
	if conf.Finishing != nil {
		finishing = NewOrFinishing(finishing, conf.Finishing)
	}

	defer algo.recover(start)

	algo.receiveStatus()

	for err == nil && algo.status.Value == Running && !IsFinished(finishing, algo) {
		select { // check for algo status update
		case status := <-algo.statusChannel:
			algo.setStatus(status, true)
			if status.Value == Idle {
				algo.ackChannel <- true
				algo.receiveStatus()
			}
		default:
			// run implementation
			// newData = algo.newData
			centroids, runtimeFigures, err = algo.impl.Iterate(
				NewSimpleOCModel(
					algo.conf, algo.space, algo.status, algo.runtimeFigures, algo.centroids,
				),
			)
			duration = time.Now().Sub(start)
			if err == nil {
				if centroids != nil { // an iteration has been executed
					algo.modelMutex.Lock()
					algo.iterations++
					algo.saveIterContext(
						centroids, runtimeFigures, duration,
					)
					algo.modelMutex.Unlock()
				}
				// temporize iteration
				if iterFreq > 0 { // with iteration freqency
					var diff = iterFreq - time.Now().Sub(lastIterationTime)
					time.Sleep(diff)
					lastIterationTime = time.Now()
				}
			}
		}
	}
	if err == nil {
		if algo.status.Value == Running {
			algo.setStatus(NewOCStatus(Ready), true)
		}
	} else {
		algo.setStatus(NewOCStatusError(err), true)
	}
}

func (algo *Algo) updateRuntimeFigures() {
	algo.runtimeFigures[Iterations] = float64(algo.iterations)
	algo.runtimeFigures[PushedData] = float64(algo.pushedData)
	algo.runtimeFigures[LastDataTime] = float64(algo.lastDataTime)
}

func (algo *Algo) saveIterContext(centroids Clust, runtimeFigures RuntimeFigures, duration time.Duration) {
	if runtimeFigures == nil {
		runtimeFigures = RuntimeFigures{}
	}
	runtimeFigures[Duration] += float64(duration)
	algo.centroids = centroids
	algo.runtimeFigures = runtimeFigures
	algo.updateRuntimeFigures()
}

/*
// SetConf algo configuration and space
func (algo *Algo) SetConf(conf Conf) (err error) {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	if algo.status.Value == Running {
		err = ErrRunning
	} else {
		err = PrepareConf(conf)
		if err == nil {
			impl, err := algo.impl.Copy(algo)
			if err == nil {
				algo.modelMutex.Lock()
				algo.impl = impl
				algo.conf = conf
				algo.modelMutex.Unlock()
			} else {
				algo.setStatus(NewOCStatusError(err), true)
			}
		}
	}
	return
}

// SetSpace algo configuration and space
func (algo *Algo) SetSpace(space Space) (err error) {
	algo.ctrlMutex.Lock()
	defer algo.ctrlMutex.Unlock()
	if algo.status.Value == Running {
		err = ErrRunning
	} else {
		impl, err := algo.impl.Copy(algo)
		if err == nil {
			algo.modelMutex.Lock()
			algo.impl = impl
			algo.space = space
			algo.modelMutex.Unlock()
		} else {
			algo.setStatus(NewOCStatusError(err), true)
		}
	}
	return
}*/

// Copy make a copy of this algo with new conf and space
func (algo *Algo) Copy(conf Conf, space Space) (oc OnlineClust, err error) {
	impl, err := algo.impl.Copy(algo)
	if err == nil {
		oc = NewAlgo(conf, impl, space)
	}
	return
}
