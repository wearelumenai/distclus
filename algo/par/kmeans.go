package par

import (
	"distclus/algo"
	"distclus/core"
	"sync"
	"runtime"
)

// message exchanged between kmeans go routines, actually weighted means
type msgKMeans struct {
	// the mean for a subset of elements
	dba core.Elemt
	// the number of elements that participate to the mean
	card int
}

// aggAssign receives a partition of elements, assign them to a cluster and update the mean for this cluster.
// when partition is exhausted, send the means and their cardinality to out channel
func aggAssign(clust core.Clust, sp core.Space, elmts []core.Elemt, out chan<- []msgKMeans, wg *sync.WaitGroup) {
	defer wg.Done()

	var we = make([]msgKMeans, len(clust))

	for i := range elmts {
		var _, ix, _ = clust.Assign(elmts[i], sp)

		if we[ix].card == 0 {
			// first time we see this cluster, just copy the element
			we[ix].dba = sp.Copy(elmts[i])
			we[ix].card = 1
		} else {
			// combine for dba
			sp.Combine(we[ix].dba, we[ix].card, elmts[i], 1)
			we[ix].card += 1
		}
	}

	// send computed aggregates
	out <- we
}

// aggDBA receives partitioned weighted means from channel in and reduce them into a single mean for each cluster
// when finished send the result to out channel.
func aggDBA(sp core.Space, in <-chan []msgKMeans) []msgKMeans {
	var aggregate []msgKMeans
	for we := range in {
		if aggregate == nil {
			// first message, just take it as current aggregate
			aggregate = we
		} else {
			// combine subsequent messages to aggregate
			for i := 0; i < len(aggregate); i++ {
				switch {
				case aggregate[i].card == 0:
					// first time we see this cluster, just take the element
					aggregate[i] = we[i]

				case we[i].card > 0:
					// combine subsequent elements for this cluster
					sp.Combine(aggregate[i].dba, aggregate[i].card, we[i].dba, we[i].card)
					aggregate[i].card += we[i].card
				}
			}
		}
	}

	return aggregate
}

// Implement the KMeansSupport interface for parallel processing
type ParKMeansSupport struct {
	space core.Space
}

// Iterate implements a parallel kmeans iteration.
// It starts as many worker routines as available CPU to execute aggAssign and fan out all data to them.
// When all workers finish it aggregates partial results and compute global result.
func (support ParKMeansSupport) Iterate(km algo.KMeans, clust core.Clust) core.Clust {
	// channels
	var degree = runtime.NumCPU()
	var length = len(km.Data)
	var offset = (length-1)/degree + 1
	var out = make(chan []msgKMeans, degree)
	var wg = &sync.WaitGroup{}

	// start workers
	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var start = i * offset
		var end = start + offset

		if end > length {
			end = length
		}

		var part = (km.Data)[start:end]
		go aggAssign(clust, km.Space, part, out, wg)
	}

	// wait all workers to shutdown
	wg.Wait()
	// close the partial results channel before computing its content
	close(out)

	// read and build the result
	var aggr = aggDBA(km.Space, out)

	var result = make(core.Clust, len(aggr))
	for i := 0; i < len(clust); i++ {
		if aggr[i].card > 0 {
			result[i] = aggr[i].dba
		} else {
			result[i] = clust[i]
		}
	}

	return result
}

// NewKMeans create a new parallel KMeans algorithm instance.
func NewKMeans(conf algo.KMeansConf, initializer algo.Initializer, data []core.Elemt) algo.KMeans {
	var km = algo.NewKMeans(conf, initializer, data)
	km.KMeansSupport = ParKMeansSupport{space: conf.Space}
	return km
}
