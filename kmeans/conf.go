package kmeans

import (
	"fmt"
	"runtime"
	"time"

	"golang.org/x/exp/rand"
)

// Conf of KMeans
type Conf struct {
	Par       bool
	NumCPU    int
	K         int
	Iter      int
	FrameSize int
	RGen      *rand.Rand
}

// Verify configuratio
func Verify(conf Conf) {
	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}
}

// SetConfigDefaults initializes nil configuration values
func SetConfigDefaults(conf *Conf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if conf.NumCPU == 0 {
		conf.NumCPU = runtime.NumCPU()
	}
}
