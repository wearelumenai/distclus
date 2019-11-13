package mcmc

import (
	"distclus/core"
	"fmt"
	"time"

	"golang.org/x/exp/rand"
)

// Conf is the mcmc configuration object
type Conf struct {
	core.Conf
	Par            bool
	InitK          int // number of initial number of clusters
	FrameSize      int
	RGen           *rand.Rand
	B, Amp, R      float64
	Norm           float64
	MaxK           int
	ProbaK         []float64
	lamb, l2b, tau float64
}

// SetConfigDefaults initializes nil parameter values
func (conf *Conf) SetConfigDefaults() {
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
}

// Verify configuration parameters
func (conf *Conf) Verify() {
	conf.Conf.Verify()
	conf.SetConfigDefaults()
	if conf.InitK < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.InitK))
	}

	if conf.InitK > conf.MaxK && conf.MaxK != 0 {
		panic(fmt.Sprintf("Illegal value for Max K / Init K: %v / %v", conf.MaxK, conf.InitK))
	}
}
