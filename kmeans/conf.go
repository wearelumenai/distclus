package kmeans

import (
	"fmt"
	"runtime"
	"time"

	"go.lumenai.fr/distclus/v0/core"

	"golang.org/x/exp/rand"
)

// Conf of KMeans
type Conf struct {
	core.Conf
	Par       bool
	K         int
	FrameSize int
	RGen      *rand.Rand
	NumCPU    int // maximal number of CPU to use
}

// Verify configuratio
func (conf *Conf) Verify() {
	conf.Conf.Verify()
	conf.SetConfigDefaults()
	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}
}

// SetConfigDefaults initializes nil configuration values
func (conf *Conf) SetConfigDefaults() {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
	}
}
