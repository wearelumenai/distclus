package mcmc

import (
	"distclus/core"
	"fmt"
	"golang.org/x/exp/rand"
	"math"
)

type MCMCConf struct {
	Dim                int
	FrameSize          int
	B, Amp, R          float64
	Norm               float64
	Nu                 float64
	InitK              int
	MaxK               int
	McmcIter, InitIter int
	ProbaK             []float64
	Space              core.Space
	RGen               *rand.Rand
	lamb, l2b, tau     float64
}

func (conf *MCMCConf) Tau() float64 {
	if conf.tau == 0 {
		conf.tau = 1 / math.Sqrt(float64(conf.FrameSize*20))
	}
	return conf.tau
}

func (c *MCMCConf) L2B() float64 {
	if c.l2b == 0 {
		c.l2b = math.Log(2 * c.B)
	}
	return c.l2b
}

func (conf *MCMCConf) Lambda() float64 {
	if conf.lamb == 0 {
		var r = conf.R

		if r == 0 {
			r = 1
		}

		conf.lamb = conf.Amp * math.Sqrt(float64(conf.Dim+3)/float64(conf.FrameSize))
	}

	return conf.lamb
}

func (conf *MCMCConf) Verify() {
	if conf.InitK < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.InitK))
	}

	if conf.InitK > conf.MaxK && conf.MaxK != 0 {
		panic(fmt.Sprintf("Illegal value for Max K / Init K: %v / %v", conf.MaxK, conf.InitK))
	}

	if conf.McmcIter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.McmcIter))
	}
}