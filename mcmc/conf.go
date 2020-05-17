package mcmc

import (
	"fmt"
	"runtime"
	"time"

	"github.com/wearelumenai/distclus/core"

	"golang.org/x/exp/rand"
)

// Conf is the mcmc configuration object
type Conf struct {
	core.CtrlConf
	Par            bool
	InitK          int // number of initial number of clusters
	RGen           *rand.Rand
	B, Amp, R      float64
	Norm           float64
	MaxK           int
	ProbaK         []float64
	lamb, l2b, tau float64
	FrameSize      int
	NumCPU         int // maximal number of CPU to use
}

// SetDefaultValues initializes nil parameter values
func (conf *Conf) SetDefaultValues() {
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
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
	}
}

// Verify configuration parameters
func (conf *Conf) Verify() (err error) {
	conf.SetDefaultValues()
	if conf.InitK < 1 {
		err = fmt.Errorf("Illegal value for K: %v", conf.InitK)
	}
	if err == nil && conf.InitK > conf.MaxK && conf.MaxK != 0 {
		err = fmt.Errorf("Illegal value for Max K / Init K: %v / %v", conf.MaxK, conf.InitK)
	}
	return
}
