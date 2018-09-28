package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"runtime"
	"sync"
)

func NewParMCMC(conf MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	var mcmc = NewSeqMCMC(conf, distrib, initializer, data)
	var strategy = ParMCMCStrategy{}
	strategy.Buffer = mcmc.data
	strategy.Config = mcmc.config
	strategy.Degree = runtime.NumCPU()
	mcmc.strategy = &strategy
	return mcmc
}

type ParMCMCStrategy struct {
	Config MCMCConf
	Buffer *core.DataBuffer
	Degree int
}

type workerSupport struct {
	ParMCMCStrategy
	out chan msgMCMC
	wg  *sync.WaitGroup
}

type msgMCMC struct {
	sum  float64
	card int
}

func (strategy ParMCMCStrategy) Iterate(clust core.Clust, iter int) core.Clust {
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

func (strategy *ParMCMCStrategy) Loss(clust core.Clust) float64 {
	var workers = strategy.startMCMCWorkers(clust)
	var aggr = workers.lossAggregate()
	return aggr.sum
}

func (strategy *ParMCMCStrategy) startMCMCWorkers(clust core.Clust) workerSupport {
	var offset = (len(strategy.Buffer.Data)-1)/strategy.Degree + 1
	var workers = workerSupport{}
	workers.ParMCMCStrategy = *strategy
	workers.out = make(chan msgMCMC, strategy.Degree)
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

	var reduced msgMCMC
	reduced.sum = clust.Loss(elemts, strategy.Config.Space, strategy.Config.Norm)
	reduced.card = len(elemts)

	strategy.out <- reduced
}

func (strategy *workerSupport) lossAggregate() msgMCMC {
	var aggregate msgMCMC
	for agg := range strategy.out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
