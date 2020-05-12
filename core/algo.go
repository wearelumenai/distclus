// Package core proposes a generic framework that executes online clustering algorithm.
package core

import (
	"sync"
	"time"

	"github.com/wearelumenai/distclus/figures"
)

// OnlineClust interface
// When a prediction is made, the element can be pushed to the model.
// A prediction consists in a centroid and a label.
// The following constraints must be met (otherwise an error is returned) :
// an element can't be pushed if the algorithm is stopped,
// a prediction can't be done before the algorithm is run,
// no centroid can be returned before the algorithm is run.
type OnlineClust interface {
	OCCtrl
	OCModel
}

// Algo in charge of algorithm execution with both implementation and user configuration
type Algo struct {
	conf            Conf
	impl            Impl
	space           Space
	centroids       Clust
	status          OCStatus
	statusChannel   chan OCStatus
	ackChannel      chan bool
	runtimeFigures  figures.RuntimeFigures
	newData         int
	pushedData      int
	totalIterations int
	iterToRun       int // specific number of iterations to do
	duration        time.Duration
	lastDataTime    int64
	succeedOnce     bool
	timeout         Timeout

	ctrl  sync.RWMutex // algo controller mutex
	model sync.RWMutex // algo model mutex
}

// NewAlgo creates a new algorithm instance
func NewAlgo(conf Conf, impl Impl, space Space) (algo *Algo) {
	var ctrlConf = conf.Ctrl()
	ctrlConf.SetDefaultValues()
	ctrlConf.Verify()

	algo = &Algo{
		conf:          conf,
		impl:          impl,
		space:         space,
		status:        OCStatus{Value: Created},
		statusChannel: make(chan OCStatus),
		ackChannel:    make(chan bool),
	}

	return
}

// change of status
func (algo *Algo) setStatus(status OCStatus) {
	algo.model.Lock()
	algo.status = status
	var statusNotifier = algo.conf.Ctrl().StatusNotifier
	algo.model.Unlock()
	if statusNotifier != nil {
		go statusNotifier(algo, status)
	}
}

// change of status
func (algo *Algo) setConcurrentStatus(status OCStatus) {
	algo.ctrl.Lock()
	algo.setStatus(status)
	algo.ctrl.Unlock()
}

// receiveStatus status from main routine
func (algo *Algo) receiveStatus() {
	var status = <-algo.statusChannel
	algo.setStatus(status)
	algo.ackChannel <- true
}

// sendStatus status to run go routine
func (algo *Algo) sendStatus(status OCStatus) (ok bool) {
	algo.statusChannel <- status
	_, ok = <-algo.ackChannel
	return
}
