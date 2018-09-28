package kmeans

import (
	"distclus/core"
)

type KMeans struct {
	template    *core.AlgorithmTemplate
	data        *core.DataBuffer
	config      KMeansConf
	strategy    KMeansStrategy
	initializer core.Initializer
}

type KMeansStrategy interface {
	Iterate(proposal core.Clust) core.Clust
}

func (km *KMeans) Centroids() (c core.Clust, err error) {
	return km.template.Centroids()
}

func (km *KMeans) Push(elemt core.Elemt) (err error) {
	return km.template.Push(elemt)
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return km.template.Predict(elemt, push)
}

func (km *KMeans) Run(async bool) {
	km.template.Run(async)
}

func (km *KMeans) Close() {
	km.template.Close()
}

func (km *KMeans) initializeAlgorithm() (centroids core.Clust, ready bool) {
	km.data.Apply()
	return km.initializer(km.config.K, km.data.Data, km.config.Space, km.config.RGen)

}

func (km *KMeans) runAlgorithm(closing <-chan bool) {
	for iter, loop := 0, true; iter < km.config.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false

		default:
			km.template.Clust = km.strategy.Iterate(km.template.Clust)
			km.data.Apply()
		}
	}
}
