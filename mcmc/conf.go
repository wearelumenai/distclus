package mcmc

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"golang.org/x/exp/rand"
)

// Conf is the mcmc configuration object
type Conf struct {
	Par            bool
	Iter           int
	InitK          int
	FrameSize      int
	RGen           *rand.Rand
	B, Amp, R      float64
	Norm           float64
	MaxK           int
	McmcIter       int
	Timeout        int
	ProbaK         []float64
	lamb, l2b, tau float64
	NumCPU         int
}

// SetConfigDefaults initializes nil parameter values
func SetConfigDefaults(conf *Conf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if len(conf.ProbaK) == 0 {
		conf.ProbaK = []float64{1, 8, 1}
	}
	if conf.Norm == 0 {
		conf.Norm = 2
	}
	if conf.MaxK == 0 {
		conf.MaxK = 16
	}
	if conf.B == 0 {
		conf.B = 1
	}
	if conf.Iter == 0 {
		conf.Iter = 1
	}
	if conf.Timeout == 0 {
		conf.Timeout = math.MaxInt64
	}
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
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
