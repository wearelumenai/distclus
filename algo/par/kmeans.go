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
	var aggr = dbaReduce(km.Space, out)
	return buildResult(clust, aggr)
}

func startKMeansWorkers(km algo.KMeans, clust core.Clust) (chan []msgKMeans) {
	var degree = runtime.NumCPU()
	var offset = (len(km.Data)-1)/degree + 1
	var out = make(chan []msgKMeans, degree)
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

func buildResult(clust core.Clust, aggr []msgKMeans) core.Clust {
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

// message exchanged between kmeans go routines, actually weighted means
type msgKMeans struct {
	// the mean for a subset of elements
	dba core.Elemt
	// the number of elements that participate to the mean
	card int
}

// assignMapReduce receives a partition of elements, assign them to a cluster and update the mean for this cluster.
// when partition is exhausted, send the means and their cardinality to out channel
func assignMapReduce(clust core.Clust, elmts []core.Elemt, sp core.Space, out chan<- []msgKMeans, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced = make([]msgKMeans, len(clust))

	for _, elmt := range elmts {
		reduced = assignCombine(reduced, clust, elmt, sp)
	}

	// send computed aggregates
	out <- reduced
}

func assignCombine(reduced []msgKMeans, clust core.Clust, elemt core.Elemt, sp core.Space) []msgKMeans{
	var _, ix, _ = clust.Assign(elemt, sp)

	if reduced[ix].card == 0 {
		// first time reduced see this cluster, just copy the element
		reduced[ix].dba = sp.Copy(elemt)
		reduced[ix].card = 1
	} else {
		// combine for dba
		sp.Combine(reduced[ix].dba, reduced[ix].card, elemt, 1)
		reduced[ix].card += 1
	}

	return reduced
}

// dbaReduce receives partitioned weighted means from channel in and reduce them into a single mean for each cluster
// when finished send the result to out channel.
func dbaReduce(sp core.Space, in <-chan []msgKMeans) []msgKMeans {
	var aggregate []msgKMeans
	for other := range in {
		if aggregate == nil {
			// first message, just take it as current aggregate
			aggregate = other
		} else {
			aggregate = dbaCombine(aggregate, other, sp)
		}
	}

	return aggregate
}

func dbaCombine(aggregate []msgKMeans, other []msgKMeans, sp core.Space) []msgKMeans {
	// combine subsequent messages to aggregate
	for i := 0; i < len(aggregate); i++ {
		switch {
		case aggregate[i].card == 0:
			// first time other see this cluster, just take the element
			aggregate[i] = other[i]

		case other[i].card > 0:
			// combine subsequent elements for this cluster
			sp.Combine(aggregate[i].dba, aggregate[i].card, other[i].dba, other[i].card)
			aggregate[i].card += other[i].card
		}
	}

	return aggregate
}

