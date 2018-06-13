package par

import (
	"distclus/algo"
	"distclus/core"
	"golang.org/x/exp/rand"
)

type ParKMeansSupport struct {
	space core.Space
}

func (iter ParKMeansSupport) Initialize(k int, nodes []core.Elemt, space core.Space, src *rand.Rand) algo.Clust {
	panic("implement me")
}

func (iter ParKMeansSupport) Iterate(km algo.KMeans) algo.Clust {
	var clust, _ = km.Centroids()
	var assign = clust.AssignAll(km.Data, iter.space)
	var result = make(algo.Clust, len(clust))

	for k, cluster := range assign {
		if len(cluster) != 0 {
			result[k] = algo.DBA(cluster, iter.space)
		}
	}

	return result
}


func NewKMeans(k int, iter int, space core.Space, initializer algo.Initializer) algo.KMeans {
	var km = algo.KMeans{}
	km.KMeansSupport = ParKMeansSupport{space:space}
	return km
}
