package kmeans

import (
	"distclus/core"
	"golang.org/x/exp/rand"
	"time"
)

type KMeans struct {
	*core.AbstractAlgo
	KMeansSupport
	config      KMeansConf
}

type KMeansSupport interface {
	Iterate(proposal core.Clust) core.Clust
}

func NewSeqKMeans(config KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	config.Verify()
	setConfigDefaults(&config)

	var km KMeans
	km.AbstractAlgo = core.NewAlgo(config.AlgoConf, data, initializer)
	km.AbstractAlgo.RunAlgorithm = km.runAlgorithm
	km.config = config
	km.KMeansSupport = SeqKMeansSupport{buffer: &km.Buffer, config: km.config}

	return &km
}

func setConfigDefaults(conf *KMeansConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
}

func (km *KMeans) Centroids() (c core.Clust, err error) {
	return km.AbstractAlgo.Centroids()
}

func (km *KMeans) Push(elemt core.Elemt) (err error) {
	return km.AbstractAlgo.Push(elemt)
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return km.AbstractAlgo.Predict(elemt, push)
}

func (km *KMeans) Run(async bool) {
	km.AbstractAlgo.Run(async)
}

func (km *KMeans) Close() {
	km.AbstractAlgo.Close()
}

func (km *KMeans) runAlgorithm(closing <-chan bool) {
	for iter, loop := 0, true; iter < km.config.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false

		default:
			km.Clust = km.Iterate(km.Clust)
			km.Buffer.Apply()
		}
	}
}

