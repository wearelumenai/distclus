package kmeans

import (
	"distclus/core"
	"golang.org/x/exp/rand"
	"time"
)

func NewSeqKMeans(config KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	config.Verify()
	setConfigDefaults(&config)

	var km KMeans
	km.data = core.NewBuffer(data, config.FrameSize)
	var algoTemplateMethods = core.AlgorithmTemplateMethods{
		Initialize: km.initializeAlgorithm,
		Run:        km.runAlgorithm,
	}
	km.template = core.NewAlgo(config.AlgorithmConf, km.data, algoTemplateMethods)

	km.config = config
	km.initializer = initializer
	km.strategy = SeqKMeansStrategy{Buffer: km.data, Config: km.config}

	return &km
}

type SeqKMeansStrategy struct {
	Config KMeansConf
	Buffer *core.DataBuffer
}

func setConfigDefaults(conf *KMeansConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
}

func (strategy SeqKMeansStrategy) Iterate(clust core.Clust) core.Clust {
	var result, _ = clust.AssignDBA(strategy.Buffer.Data, strategy.Config.Space)
	return strategy.buildResult(clust, result)
}

func (strategy SeqKMeansStrategy) buildResult(clust core.Clust, result core.Clust) core.Clust {
	for i := 0; i < len(result); i++ {
		if result[i] == nil {
			result[i] = clust[i]
		}
	}
	return result
}
