package kmeans

import (
	"distclus/core"
	"fmt"
)

type KMeansConf struct {
	core.AlgorithmConf
	Iter int
}

func (conf *KMeansConf) Verify() {
	if conf.InitK < 1 {
		panic(fmt.Sprintf("Illegal value for InitK: %v", conf.InitK))
	}

	if conf.Iter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.Iter))
	}
}
