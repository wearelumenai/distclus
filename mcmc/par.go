package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"runtime"
	"sync"
)

// NewParImpl returns a new parallelized algorithm implementation
func NewParImpl(conf Conf, distrib Distrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	var mcmc = NewSeqImpl(conf, distrib, initializer, data)
	var strategy = ParStrategy{}
	strategy.Buffer = mcmc.data
	strategy.Degree = runtime.NumCPU()
	mcmc.strategy = &strategy
	return mcmc
}

// ParStrategy defines a parallelized strategy
type ParStrategy struct {
	Degree int
}

type workerSupport struct {
	ParStrategy
	out chan msgMCMC
	wg  *sync.WaitGroup
}

type msgMCMC struct {
	sum  float64
	card int
}

// Iterate is the iterative execution
func (strategy *ParStrategy) Iterate(conf Conf, space core.Space, clust core.Clust, iter int) core.Clust {
	var kmeansConf = kmeans.Conf{
		K:    len(clust),
		Iter: iter,
	}
	var km = kmeans.NewParImpl(kmeansConf, clust.Initializer, strategy.Buffer.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

// Loss aclculates the loss distance of input centroids
func (strategy *ParStrategy) Loss(conf Conf, space core.Space, clust core.Clust) float64 {
	var workers = strategy.startWorkers(conf, space, clust)
	var aggr = workers.lossAggregate()
	return aggr.sum
}

func (strategy *ParStrategy) startWorkers(conf Conf, space core.Space, clust core.Clust) workerSupport {
	var offset = (len(strategy.Buffer.Data)-1)/strategy.Degree + 1
	var workers = workerSupport{}
	workers.ParStrategy = *strategy
	workers.out = make(chan msgMCMC, strategy.Degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(strategy.Degree)

	for i := 0; i < strategy.Degree; i++ {
		var part = core.GetChunk(i, offset, strategy.Buffer.Data)
		go workers.lossMapReduce(conf, space, clust, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return workers
}

func (strategy *workerSupport) lossMapReduce(conf Conf, space core.Space, clust core.Clust, elemts []core.Elemt) {
	defer strategy.wg.Done()

	var reduced msgMCMC
	reduced.sum = clust.Loss(elemts, space, conf.Norm)
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
