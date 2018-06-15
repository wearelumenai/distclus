package par

import (
	"distclus/algo"
	"distclus/core"
	"golang.org/x/exp/rand"
)

type ParKMeansSupport struct {
	space core.Space
}

func (support ParKMeansSupport) Initialize(k int, nodes []core.Elemt, space core.Space, src *rand.Rand) algo.Clust {
	panic("implement me")
}

func (support ParKMeansSupport) Iterate(km algo.KMeans, clust algo.Clust) algo.Clust {
	var assign = clust.AssignAll(km.Data, support.space)
	var result = make(algo.Clust, len(clust))

	for k, cluster := range assign {
		if len(cluster) != 0 {
			result[k], _ = algo.DBA(cluster, support.space)
		}
	}

	return result
}

func NewKMeans(conf algo.KMeansConf, initializer algo.Initializer) algo.KMeans {
	var km = algo.NewKMeans(conf, initializer)
	km.KMeansSupport = ParKMeansSupport{space: conf.Space}
	return km
}
