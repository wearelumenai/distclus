package kmeans

import (
	"fmt"
	"runtime"
	"time"

	"github.com/wearelumenai/distclus/core"

	"golang.org/x/exp/rand"
)

// Conf of KMeans
type Conf struct {
	core.CtrlConf
	Par       bool
	K         int
	FrameSize int
	RGen      *rand.Rand
	NumCPU    int // maximal number of CPU to use
}

// Verify configuratio
func (conf *Conf) Verify() (err error) {
	conf.SetDefaultValues()
	if conf.K < 1 {
		err = fmt.Errorf("Illegal value for K: %v", conf.K)
	}
	return
}

// SetDefaultValues initializes nil configuration values
func (conf *Conf) SetDefaultValues() {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
	}
}
