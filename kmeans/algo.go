package kmeans

import (
	"distclus/core"
)

// KMeans algorithm abstract implementation
type KMeans struct {
	template    *core.AlgorithmTemplate
	data        *core.DataBuffer
	config      KMeansConf
	strategy    KMeansStrategy
	initializer core.Initializer
}

// Abstract KMeans strategy to be implemented by concrete algorithms
type KMeansStrategy interface {
	Iterate(proposal core.Clust) core.Clust
}

// Get the centroids currently found by the algorithm
func (km *KMeans) Centroids() (c core.Clust, err error) {
	return km.template.Centroids()
}

// Push a new observation in the algorithm
func (km *KMeans) Push(elemt core.Elemt) (err error) {
	return km.template.Push(elemt)
}

// Predict the cluster for a new observation
func (km *KMeans) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return km.template.Predict(elemt, push)
}

// Run the algorithm, asynchronously if async is true
func (km *KMeans) Run(async bool) {
	km.template.Run(async)
}

// Stop the algorithm
func (km *KMeans) Close() {
	km.template.Close()
}

// Algorithm first iteration centroids initialization
func (km *KMeans) initializeAlgorithm() (centroids core.Clust, ready bool) {
	km.data.Apply()
	return km.initializer(km.config.K, km.data.Data, km.config.Space, km.config.RGen)
}

// run the algorithm until signal received on closing channel or iteration number is reached
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
