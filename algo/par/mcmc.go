package par

import (
	"distclus/algo"
	"distclus/core"
	"runtime"
	"sync"
)

func NewMCMC(conf algo.MCMCConf, distrib algo.MCMCDistrib, initializer core.Initializer, data []core.Elemt) algo.MCMC {
	var mcmc = algo.NewMCMC(conf, distrib, initializer, data)

	mcmc.MCMCSupport = ParMCMCSupport{conf}

	return mcmc
}

type ParMCMCSupport struct {
	config algo.MCMCConf
}

func (supp ParMCMCSupport) Iterate(m algo.MCMC, clust core.Clust, iter int) core.Clust {
	var conf = algo.KMeansConf{K: len(clust), Iter: iter, Space: supp.config.Space}
	var km = NewKMeans(conf, clust.Initializer, m.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

func (supp ParMCMCSupport) Loss(m algo.MCMC, clust core.Clust) float64 {
	var out = startMCMCWorkers(m, clust)
	var aggr = lossAggregate(out)
	return aggr.sum
}

func startMCMCWorkers(m algo.MCMC, clust core.Clust) (chan msgMCMC) {
	var degree = runtime.NumCPU()
	var offset = (len(m.Data)-1)/degree + 1
	var out = make(chan msgMCMC, degree)
	var wg = &sync.WaitGroup{}

	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var part = getChunk(i, offset, m.Data)
		go lossMapReduce(clust, part, m.Space, m.Norm, out, wg)
	}

	wg.Wait()
	close(out)

	return out
}

type msgMCMC struct {
	sum float64
	card int
}

func lossMapReduce(clust core.Clust, elemts []core.Elemt, space core.Space, norm float64, out chan<- msgMCMC, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced msgMCMC
	reduced.sum = clust.Loss(elemts, space, norm)
	reduced.card = len(elemts)

	out <- reduced
}

func lossAggregate(out chan msgMCMC) msgMCMC {
	var aggregate msgMCMC
	for agg := range out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
