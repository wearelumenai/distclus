// Package core proposes a generic framework that executes online clustering algorithm.
package core

import (
	"sync"
	"time"
)

// OnlineClust interface
// When a prediction is made, the element can be pushed to the model.
// A prediction consists in a centroid and a label.
type OnlineClust interface {
	OCCtrl
	OCModel
}

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	conf           Conf
	impl           Impl
	space          Space
	centroids      Clust
	status         OCStatus
	statusChannel  chan OCStatus
	ackChannel     chan bool
	runtimeFigures RuntimeFigures
	newData        int
	pushedData     int
	iterations     int
	duration       time.Duration
	lastDataTime   int64
	timeout        Timeout

	modelMutex  sync.RWMutex // algo model mutex
	statusMutex sync.RWMutex // algo model mutex
}

// NewAlgo creates a new algorithm instance
func NewAlgo(conf Conf, impl Impl, space Space) (algo *Algo) {
	var err = PrepareConf(conf)

	if err != nil {
		panic(err)
	}

	algo = &Algo{
		conf:           conf,
		impl:           impl,
		space:          space,
		status:         OCStatus{Value: Created},
		statusChannel:  make(chan OCStatus),
		ackChannel:     make(chan bool),
		runtimeFigures: RuntimeFigures{},
	}

	return
}

// change of status
func (algo *Algo) setStatus(status OCStatus, safe bool) {
	if safe {
		algo.statusMutex.Lock()
		algo.status = status
		algo.statusMutex.Unlock()
	} else {
		algo.status = status
	}
	algo.modelMutex.RLock()
	var statusNotifier = algo.conf.Ctrl().StatusNotifier
	algo.modelMutex.RUnlock()
	if statusNotifier != nil {
		go statusNotifier(algo, status)
	}
}

// receiveStatus status from main routine
func (algo *Algo) receiveStatus() {
	var status = <-algo.statusChannel
	algo.setStatus(status, true)
	algo.ackChannel <- true
}

// sendStatus status to run go routine
func (algo *Algo) sendStatus(status OCStatus) (ok bool) {
	algo.statusChannel <- status
	_, ok = <-algo.ackChannel
	return
}
