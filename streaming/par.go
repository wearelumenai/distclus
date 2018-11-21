package streaming

import (
	"distclus/core"
	"distclus/kmeans"
	"runtime"
	"sync"
)

func NewParStreaming(conf StreamingConf, distrib StreamingDistrib, initializer core.Initializer, data []core.Elemt) *Streaming {
	var streaming = NewSeqStreaming(conf, distrib, initializer, data)
	var strategy = ParStreamingStrategy{}
	strategy.Buffer = streaming.data
	strategy.Config = streaming.config
	strategy.Degree = runtime.NumCPU()
	streaming.strategy = &strategy
	return streaming
}

type ParStreamingStrategy struct {
	Config StreamingConf
	Buffer *core.DataBuffer
	Degree int
}

type workerSupport struct {
	ParStreamingStrategy
	out chan msgStreaming
	wg  *sync.WaitGroup
}

type msgStreaming struct {
	sum  float64
	card int
}

func (strategy ParStreamingStrategy) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{
		AlgorithmConf: core.AlgorithmConf{
			Space: strategy.Config.Space,
		},
		K:    len(clust),
		Iter: iter,
	}
	var km = kmeans.NewParKMeans(conf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

func (strategy *ParStreamingStrategy) Loss(clust core.Clust) float64 {
	var workers = strategy.startStreamingWorkers(clust)
	var aggr = workers.lossAggregate()
	return aggr.sum
}

func (strategy *ParStreamingStrategy) startStreamingWorkers(clust core.Clust) workerSupport {
	var offset = (len(strategy.Buffer.Data)-1)/strategy.Degree + 1
	var workers = workerSupport{}
	workers.ParStreamingStrategy = *strategy
	workers.out = make(chan msgStreaming, strategy.Degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(strategy.Degree)

	for i := 0; i < strategy.Degree; i++ {
		var part = core.GetChunk(i, offset, strategy.Buffer.Data)
		go workers.lossMapReduce(clust, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return workers
}

func (strategy *workerSupport) lossMapReduce(clust core.Clust, elemts []core.Elemt) {
	defer strategy.wg.Done()

	var reduced msgStreaming
	reduced.sum = clust.Loss(elemts, strategy.Config.Space, strategy.Config.Norm)
	reduced.card = len(elemts)

	strategy.out <- reduced
}

func (strategy *workerSupport) lossAggregate() msgStreaming {
	var aggregate msgStreaming
	for agg := range strategy.out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
