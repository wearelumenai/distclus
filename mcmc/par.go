package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"runtime"
	"sync"
)

func NewParMCMC(conf MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	var algo = NewSeqMCMC(conf, distrib, initializer, data)
	var support = ParMCMCSupport{}
	support.buffer = &algo.Buffer
	support.config = algo.config
	support.degree = runtime.NumCPU()
	algo.MCMCSupport = &support
	return algo
}

type ParMCMCSupport struct {
	config MCMCConf
	buffer *core.Buffer
	degree int
}

type workerSupport struct {
	ParMCMCSupport
	out chan msgMCMC
	wg *sync.WaitGroup
}

type msgMCMC struct {
	sum  float64
	card int
}

func (support ParMCMCSupport) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{K: len(clust), Iter: iter, Space: support.config.Space}
	var km = kmeans.NewParKMeans(conf, clust.Initializer, support.buffer.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

func (support *ParMCMCSupport) Loss(clust core.Clust) float64 {
	var workers = support.startMCMCWorkers(clust)
	var aggr = workers.lossAggregate()
	return aggr.sum
}

func (support *ParMCMCSupport) startMCMCWorkers(clust core.Clust) workerSupport  {
	var offset = (len(support.buffer.Data)-1)/support.degree + 1
	var workers = workerSupport{	}
	workers.ParMCMCSupport = *support
	workers.out = make(chan msgMCMC, support.degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(support.degree)

	for i := 0; i < support.degree; i++ {
		var part = getChunk(i, offset, support.buffer.Data)
		go workers.lossMapReduce(clust, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return  workers
}

func getChunk(i int, offset int, elemts []core.Elemt) []core.Elemt {
	var start = i * offset
	var end = start + offset

	if end > len(elemts) {
		end = len(elemts)
	}

	return elemts[start:end]
}

func (support *workerSupport) lossMapReduce(clust core.Clust, elemts []core.Elemt) {
	defer support.wg.Done()

	var reduced msgMCMC
	reduced.sum = clust.Loss(elemts, support.config.Space, support.config.Norm)
	reduced.card = len(elemts)

	support.out <- reduced
}

func (support *workerSupport) lossAggregate() msgMCMC {
	var aggregate msgMCMC
	for agg := range support.out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
