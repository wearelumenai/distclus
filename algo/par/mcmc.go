package par

import (
	"distclus/algo"
	"distclus/core"
	"math"
	"runtime"
	"sync"
)

// Implement the MCMCSupport interface for parallel processing
type ParMCMCSupport struct {
	config algo.MCMCConf
}

// Iterate MCMC in parallel by calling the parallel kmeans implementation
func (supp ParMCMCSupport) Iterate(m algo.MCMC, clust algo.Clust, iter int) algo.Clust {
	var conf = algo.KMeansConf{K: len(clust), Iter: iter, Space: supp.config.Space}
	var km = NewKMeans(conf, clust.Initializer, m.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

// message exchanged between kmeans go routines, actually weighted means
type msgMCMC struct {
	// the loss sum for a subset of elements
	sum float64
	// the number of elements that participate to the loss
	card int
}

// aggElemntLoss receives elements from in channel, compute their participation to the global loss
// when in is closed, send the partial loss and the corresponding cardinality to out channel
func aggElemtLoss(clust algo.Clust, space core.Space, norm float64, elmts []core.Elemt, out chan<- msgMCMC, wg *sync.WaitGroup) {
	defer wg.Done()

	var msg msgMCMC
	for i := range elmts {
		var min = space.Dist(elmts[i], clust[0])
		for j := 1; j < len(clust); j++ {
			// find the cluster and the minimal distance
			var d = space.Dist(elmts[i], clust[j])
			if min > d {
				min = d
			}
		}

		msg.sum += math.Pow(min, norm)
		msg.card += 1
	}

	out <- msg
}

// Loss compute the loss in parallel.
// It starts as many worker routines as available CPU to execute aggElmtLoss and fan out all data to them.
// When all workers finish it aggregate partial losses and compute global loss.
func (supp ParMCMCSupport) Loss(m algo.MCMC, proposal algo.Clust) float64 {
	// channels
	var degree = runtime.NumCPU()
	var length = len(m.Data)
	var offset = (length-1)/degree + 1
	var out = make(chan msgMCMC, degree)
	var wg = &sync.WaitGroup{}

	// start workers
	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var start = i * offset
		var end = start + offset

		if end > length {
			end = length
		}

		var part = m.Data[start:end]
		go aggElemtLoss(proposal, m.Space, m.Norm, part, out, wg)
	}

	// wait all workers to shutdown
	wg.Wait()
	// close the partial losses channel before computing its content
	close(out)

	// read and build the result
	var aggr = aggLoss(out)

	return aggr.sum
}

func aggLoss(out chan msgMCMC) msgMCMC {
	var aggregate msgMCMC
	for agg := range out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}

// NewMCMC create a new parallel MCMC algorithm instance.
func NewMCMC(conf algo.MCMCConf, distrib algo.MCMCDistrib, initializer algo.Initializer, data []core.Elemt) algo.MCMC {
	var mcmc = algo.NewMCMC(conf, distrib, initializer, data)

	mcmc.MCMCSupport = ParMCMCSupport{conf}

	return mcmc
}
