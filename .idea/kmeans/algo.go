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
func (km *KMeans) Centroids() (core.Clust, error) {
	return km.template.Centroids()
}

// Push a new observation in the algorithm
func (km *KMeans) Push(elemt core.Elemt) error {
	return km.template.Push(elemt)
}

// Predict the cluster for a new observation
func (km *KMeans) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
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

func NewKMeans(config KMeansConf, initializer core.Initializer, data []core.Elemt) (km *KMeans) {
	config.Verify()
	setConfigDefaults(&config)

	var buffer = core.NewDataBuffer(data, config.FrameSize)
	km = &KMeans{
		data:        buffer,
		config:      config,
		initializer: initializer,
		strategy:    SeqKMeansStrategy{Buffer: buffer, Config: config},
	}
	var algoTemplateMethods = core.AlgorithmTemplateMethods{
		Initialize: km.initializeAlgorithm,
		Run:        km.runAlgorithm,
	}
	km.template = core.NewAlgorithmTemplate(config.AlgorithmConf, km.data, algoTemplateMethods)

	return
}

// Algorithm first iteration centroids initialization
func (km *KMeans) initializeAlgorithm() (core.Clust, bool) {
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
