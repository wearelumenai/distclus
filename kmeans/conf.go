package kmeans

import (
	"distclus/core"
	"fmt"
	"golang.org/x/exp/rand"
)

type KMeansConf struct {
	K     int
	Iter  int
	Space core.Space
	RGen  *rand.Rand
}

func (conf *KMeansConf) Verify() {
	if conf.K < 1 {
		panic(fmt.Sprintf("Illegal value for InitK: %v", conf.K))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}
}
