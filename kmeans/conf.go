package kmeans

import (
	"distclus/core"
	"fmt"
	"golang.org/x/exp/rand"
)

type KMeansConf struct {
	core.AlgorithmConf
	K         int
	Iter      int
	FrameSize int
	RGen      *rand.Rand
}

func (conf *KMeansConf) Verify() {
	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.K))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}
}
