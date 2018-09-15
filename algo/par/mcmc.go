package par

import (
	"distclus/algo"
	"distclus/core"
	"math"
	"runtime"
	"sync"
)

// NewMCMC create a new parallel MCMC algorithm instance.
func NewMCMC(conf algo.MCMCConf, distrib algo.MCMCDistrib, initializer core.Initializer, data []core.Elemt) algo.MCMC {
	var mcmc = algo.NewMCMC(conf, distrib, initializer, data)

	mcmc.MCMCSupport = ParMCMCSupport{conf}

	return mcmc
}

// Implement the MCMCSupport interface for parallel processing
type ParMCMCSupport struct {
	config algo.MCMCConf
}

// Iterate MCMC in parallel by calling the parallel kmeans implementation
func (supp ParMCMCSupport) Iterate(m algo.MCMC, clust core.Clust, iter int) core.Clust {
	var conf = algo.KMeansConf{K: len(clust), Iter: iter, Space: supp.config.Space}
	var km = NewKMeans(conf, clust.Initializer, m.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

// Loss compute the loss in parallel.
// It starts as many worker routines as available CPU to execute aggElmtLoss and fan out all data to them.
// When all workers finish it aggregate partial losses and compute global loss.
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

// message exchanged between kmeans go routines, actually weighted means
type msgMCMC struct {
	// the loss sum for a subset of elements
	sum float64
	// the number of elements that participate to the loss
	card int
}

// lossMapReduce receives elements from in channel, compute their participation to the global loss
// when in is closed, send the partial loss and the corresponding cardinality to out channel
func lossMapReduce(clust core.Clust, elmts []core.Elemt, space core.Space, norm float64, out chan<- msgMCMC, wg *sync.WaitGroup) {
	defer wg.Done()

	var msg msgMCMC
	for _, elemt := range elmts {
		msg = lossCombine(msg, elemt, clust, space, norm)
	}

	out <- msg
}

func lossCombine(msg msgMCMC, elemt core.Elemt, clust core.Clust, space core.Space, norm float64) msgMCMC {
	var min = space.Dist(elemt, clust[0])
	for j := 1; j < len(clust); j++ {
		// find the cluster and the minimal distance
		var d = space.Dist(elemt, clust[j])
		if min > d {
			min = d
		}
	}

	msg.sum += math.Pow(min, norm)
	msg.card += 1

	return msg
}

func lossAggregate(out chan msgMCMC) msgMCMC {
	var aggregate msgMCMC
	for agg := range out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
