package core

import (
	"errors"
	"time"
)

// Conf is implementation configuration interface
type Conf interface {
	Verify()
	Ctrl() *CtrlConf // get controller configuration
	SetDefaultValues()
}

// CtrlConf specific to algo controller
type CtrlConf struct {
	Iter           int            // minimal number of iteration before sleeping. Default unlimited
	IterFreq       float64        // maximal number of iteration per seconds
	Timeout        time.Duration  // minimal number of nanoseconds before stopping the algorithm
	DataPerIter    int            // minimal pushed data number before iterating
	IterPerData    int            // minimal iterations per `DataPerIter` data
	StatusNotifier StatusNotifier // algo execution notifier
	Finishing      Finishing      // algo convergence matcher
}

// Verify conf parameters
func (conf *CtrlConf) Verify() {
	conf.SetDefaultValues()
	if conf.IterPerData < 0 {
		panic(errors.New("IterPerData must be greater or equal than 1"))
	}
	if conf.DataPerIter < 0 {
		panic(errors.New("DataPerIter must be greater or equal than 1"))
	}
	if conf.IterFreq < 0 {
		panic(errors.New("Iteration frequency must be greater or equal than 0"))
	}
	if conf.Timeout < 0 {
		panic(errors.New("Timeout must be greater or equal than 0"))
	}
	if conf.Iter < 0 {
		panic(errors.New("Iter must be greater or equal than 0"))
	}
}

// Ctrl Get pointer to algoconf
func (conf *CtrlConf) Ctrl() *CtrlConf {
	return conf
}

// SetDefaultValues set
func (conf *CtrlConf) SetDefaultValues() {
	if conf.Finishing == nil && conf.Iter > 0 {
		conf.Finishing = IterationsFinishing{
			MaxIter: conf.Iter,
		}
	}
}
