package core

import (
	"errors"
	"time"
)

// Conf is implementation configuration interface
type Conf interface {
	Verify() error
	Ctrl() *CtrlConf // get controller configuration
	SetDefaultValues()
}

// CtrlConf specific to algo controller
type CtrlConf struct {
	Iter           int            // minimal number of iteration before sleeping. Default unlimited
	IterFreq       float64        // maximal number of iteration per seconds
	Timeout        time.Duration  // minimal number of nanoseconds before stopping the algorithm. 0 is infinite (default)
	DataPerIter    int            // minimal pushed data number before iterating
	IterPerData    int            // minimal iterations per `DataPerIter` data
	StatusNotifier StatusNotifier // algo execution notifier
	Finishing      Finishing      // algo convergence matcher
}

// Verify conf parameters
func (conf *CtrlConf) Verify() (err error) {
	conf.SetDefaultValues()
	if conf.IterPerData < 0 {
		err = errors.New("IterPerData must be greater or equal than 1")
	}
	if err == nil && conf.DataPerIter < 0 {
		err = errors.New("DataPerIter must be greater or equal than 1")
	}
	if err == nil && conf.IterFreq < 0 {
		err = errors.New("Iteration frequency must be greater or equal than 0")
	}
	if err == nil && conf.Timeout < 0 {
		err = errors.New("Timeout must be greater or equal than 0")
	}
	if err == nil && conf.Iter < 0 {
		err = errors.New("Iter must be greater or equal than 0")
	}
	return
}

// Ctrl Get pointer to algoconf
func (conf *CtrlConf) Ctrl() *CtrlConf {
	return conf
}

// SetDefaultValues set
func (conf *CtrlConf) SetDefaultValues() {
}

// PrepareConf before using it in algo
func PrepareConf(conf Conf) (err error) {
	conf.SetDefaultValues()
	conf.Ctrl().SetDefaultValues()
	err = conf.Verify()
	if err == nil {
		err = conf.Ctrl().Verify()
	}
	return
}
