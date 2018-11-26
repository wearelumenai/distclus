package mcmc

import (
	"distclus/core"
	"fmt"
	"math"
	"time"

	"golang.org/x/exp/rand"
)

// Conf is the mcmc configuration object
type Conf struct {
	Iter               int
	space              core.Space
	InitK              int
	FrameSize          int
	RGen               *rand.Rand
	Dim                int
	B, Amp, R          float64
	Norm               float64
	Nu                 float64
	MaxK               int
	McmcIter, InitIter int
	ProbaK             []float64
	lamb, l2b, tau     float64
}

// Tau returns configuration Tau
func (conf *Conf) Tau() float64 {
	if conf.tau == 0 {
		conf.tau = 1 / math.Sqrt(float64(conf.FrameSize*20))
	}
	return conf.tau
}

// L2B returns configuration L2B
func (conf *Conf) L2B() float64 {
	if conf.l2b == 0 {
		conf.l2b = math.Log(2 * conf.B)
	}
	return conf.l2b
}

// Lambda returns configuration lambda
func (conf *Conf) Lambda() float64 {
	if conf.lamb == 0 {
		var r = conf.R

		if r == 0 { // lambda = (d+2)sqrt(log T)/(2sqrt(T)r^2)
			r = 1
		}

		conf.lamb = conf.Amp * math.Sqrt(float64(conf.Dim+3)/float64(conf.FrameSize))
	}

	return conf.lamb
}

// SetConfigDefaults initializes nil parameter values
func SetConfigDefaults(conf *Conf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if len(conf.ProbaK) == 0 {
		conf.ProbaK = []float64{1, 0, 9}
	}
	if conf.MaxK == 0 {
		conf.MaxK = 16
	}
}

// Verify configuration parameters
func Verify(conf Conf) {
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
