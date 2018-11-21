package streaming

import (
	"distclus/core"
	"distclus/kmeans"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"time"
)

func NewSeqStreaming(config StreamingConf, distrib StreamingDistrib, initializer core.Initializer, data []core.Elemt) (streaming *Streaming) {
	setConfigDefaults(&config)
	config.Verify()

	streaming = &Streaming{
		data: core.NewDataBuffer(data,config.FrameSize),
		config: config,
		initializer: initializer,
		distrib: distrib,
	}
	streaming.uniform = distuv.Uniform{Max: 1, Min: 0, Src: streaming.config.RGen}
	streaming.store = NewCenterStore(streaming.data, config.Space, streaming.config.RGen)
	streaming.strategy = SeqStreamingStrategy{Buffer: streaming.data, Config: streaming.config}

	var algoTemplateMethods = core.AlgorithmTemplateMethods{
		Initialize: streaming.initializeAlgorithm,
		Run:        streaming.runAlgorithm,
	}
	streaming.template = core.NewAlgorithmTemplate(config.AlgorithmConf, streaming.data, algoTemplateMethods)

	return
}

type SeqStreamingStrategy struct {
	Config StreamingConf
	Buffer *core.DataBuffer
}

func setConfigDefaults(conf *StreamingConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if len(conf.ProbaK) == 0 {
		conf.ProbaK = []float64{1, 0, 9}
	}
	if conf.MaxK == 0 {
		conf.MaxK = 16
	}
}

func (strategy SeqStreamingStrategy) Iterate(clust core.Clust, iter int) (result core.Clust) {
	var conf = kmeans.KMeansConf{
		AlgorithmConf: core.AlgorithmConf{
			Space: strategy.Config.Space,
		},
		K:    len(clust),
		Iter: iter,
		RGen: strategy.Config.RGen,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()
	result, _ = km.Centroids()

	return
}

func (strategy SeqStreamingStrategy) Loss(proposal core.Clust) float64 {
	return proposal.Loss(strategy.Buffer.Data, strategy.Config.Space, strategy.Config.Norm)
}
