package core

import (
	"errors"
	"runtime"
)

// Conf specific to algo/space configuration
type Conf struct {
	Iter           int            // minimal number of iteration before sleeping. Default unlimited
	IterFreq       float64        // maximal number of iteration per seconds
	Timeout        int            // minimal number of nanoseconds before stopping the algorithm
	NumCPU         int            // maximal number of CPU to use
	DataPerIter    int            // minimal pushed data number before iterating
	StatusNotifier StatusNotifier // algo execution notifier
}

// Verify conf parameters
func (conf *Conf) Verify() {
	conf.setConfigDefaults()
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

// SetConfigDefaults set default parameters if not given
func (conf *Conf) setConfigDefaults() {
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
	}
}

// AlgoConf Get pointer to algoconf
func (conf *Conf) AlgoConf() *Conf {
	return conf
}
