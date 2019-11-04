package core

import (
	"errors"
	"runtime"
)

// Conf specific to algo/space configuration
type Conf struct {
	MinDataCount int
	Iter         int
	IterFreq     float64
	Timeout      float64
	NumCPU       int
}

// Verify conf parameters
func (conf Conf) Verify() {
	conf.setConfigDefaults()
	if conf.MinDataCount < 0 {
		panic(errors.New("MinDataCount must be greater or equal than 1"))
	}
	if conf.Iter < 0 {
		panic(errors.New("Iterations must be greater or equal than 1"))
	}
	if conf.IterFreq < 0 {
		panic(errors.New("Iteration frequency must be greater or equal than 0"))
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
