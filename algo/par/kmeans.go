package par

import (
	"distclus/algo"
	"distclus/core"
)

type KMeans struct {
	seq algo.KMeans
	space core.Space
}

func (km *KMeans) Push(elemt core.Elemt) {
	km.seq.Push(elemt)
}

func (km *KMeans) Centroids() (algo.Clust, error) {
	return km.seq.Centroids()
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	return km.seq.Predict(elemt, push)
}

func (km *KMeans) iterate(clust *algo.Clust) {
	var assign = clust.Assign(km.seq.Data, km.space)
	for k, cluster := range assign {
		if len(cluster) != 0 {
			(*clust)[k] = algo.DBA(cluster, km.space)
		}
	}
}

func (km *KMeans) Run() {
	km.seq.Run()
}

func (km *KMeans) Close() {
	km.seq.Close()
}

func NewKMeans(k int, iter int, space core.Space, initializer algo.Initializer) KMeans {
	var km = KMeans{}
	km.seq = algo.NewKMeans(k,iter,space,initializer)
	km.space = space
	return km
}
