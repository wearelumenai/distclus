package kmeans

import (
	"distclus/core"
	"time"

	"golang.org/x/exp/rand"
)

func NewSeqKMeans(config KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	return NewKMeans(config, initializer, data)
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
