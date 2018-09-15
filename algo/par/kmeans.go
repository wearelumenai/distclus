package par

import (
	"distclus/algo"
	"distclus/core"
	"sync"
	"runtime"
)

// NewKMeans create a new parallel KMeans algorithm instance.
func NewKMeans(conf algo.KMeansConf, initializer core.Initializer, data []core.Elemt) algo.KMeans {
	var km = algo.NewKMeans(conf, initializer, data)
	km.KMeansSupport = ParKMeansSupport{space: conf.Space}
	return km
}

// Implement the KMeansSupport interface for parallel processing
type ParKMeansSupport struct {
	space core.Space
}

// Iterate implements a parallel kmeans iteration.
// It starts as many worker routines as available CPU to execute assignMapReduce and fan out all data to them.
// When all workers finish it aggregates partial results and compute global result.
func (support ParKMeansSupport) Iterate(km algo.KMeans, clust core.Clust) core.Clust {
	var out = startKMeansWorkers(km, clust)
	var aggr = assignAggregate(km.Space, out)
	return buildResult(clust, aggr)
}

func startKMeansWorkers(km algo.KMeans, clust core.Clust) (chan msgKMeans) {
	var degree = runtime.NumCPU()
	var offset = (len(km.Data)-1)/degree + 1
	var out = make(chan msgKMeans, degree)
	var wg = &sync.WaitGroup{}

	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var part = getChunk(i, offset, km.Data)
		go assignMapReduce(clust, part, km.Space, out, wg)
	}

	wg.Wait()
	close(out)

	return out
}

func getChunk(i int, offset int, elemts []core.Elemt) []core.Elemt {
	var start = i * offset
	var end = start + offset

	if end > len(elemts) {
		end = len(elemts)
	}

	return elemts[start:end]
}

// message exchanged between kmeans go routines, actually weighted means
type msgKMeans struct {
	// the mean for a subset of elements
	dbas []core.Elemt
	// the number of elements that participate to the mean
	cards []int
}

// assignMapReduce receives a partition of elements, assign them to a cluster and update the mean for this cluster.
// when partition is exhausted, send the means and their cardinality to out channel
func assignMapReduce(clust core.Clust, elemts []core.Elemt, space core.Space, out chan<- msgKMeans, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced msgKMeans
	reduced.dbas, reduced.cards = clust.AssignDBA(elemts, space)

	out <- reduced
}

// assignAggregate receives partitioned weighted means from channel in and reduce them into a single mean for each cluster
// when finished send the result to out channel.
func assignAggregate(space core.Space, in <-chan msgKMeans) msgKMeans {
	var aggregate msgKMeans
	for other := range in {
		if aggregate.dbas == nil {
			// first message, just take it as current aggregate
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = assignCombine(aggregate, other, space)
		}
	}

	return aggregate
}

func assignCombine(aggregate msgKMeans, other msgKMeans, space core.Space) msgKMeans {
	// combine subsequent messages to aggregate
	for i := 0; i < len(aggregate.dbas); i++ {
		switch {
		case aggregate.cards[i] == 0:
			// first time other see this cluster, just take the element
			aggregate.dbas[i] = other.dbas[i]
			aggregate.cards[i] = other.cards[i]

		case other.cards[i] > 0:
			// combine subsequent elements for this cluster
			space.Combine(aggregate.dbas[i], aggregate.cards[i], other.dbas[i], other.cards[i])
			aggregate.cards[i] += other.cards[i]
		}
	}

	return aggregate
}

func buildResult(clust core.Clust, aggr msgKMeans) core.Clust {
	var result = make(core.Clust, len(aggr.dbas))
	for i := 0; i < len(clust); i++ {
		if aggr.cards[i] > 0 {
			result[i] = aggr.dbas[i]
		} else {
			result[i] = clust[i]
		}
	}
	return result
}

