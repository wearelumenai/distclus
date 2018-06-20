package par

import (
	"distclus/algo"
	"distclus/core"
	"sync"
	"runtime"
)

// Implement the KMeansSupport interface for parallel processing
type ParKMeansSupport struct {
	space core.Space
}

// message exchanged between kmeans go routines, actually weighted means
type msg struct {
	// the mean of a set of elements
	dba  core.Elemt
	// the number of elements that participate to the mean
	card int
}

// aggAssign receives element from channel in, assign them to a cluster and update the mean for this cluster.
// when channel in is closed, send the means and their cardinality to out channel
func aggAssign(clust algo.Clust, sp core.Space, in <-chan core.Elemt, out chan<- []msg, wg *sync.WaitGroup) {
	defer wg.Done()

	var we = make([]msg, len(clust))

	for elmt := range in {
		var _, ix, _ = clust.Assign(elmt, sp)

		if we[ix].card == 0 {
			// first time we see this cluster, just copy the element
			we[ix].dba = sp.Copy(elmt)
			we[ix].card = 1
		} else {
			// combine for dba
			sp.Combine(we[ix].dba, we[ix].card, elmt, 1)
			we[ix].card += 1
		}
	}

	// send computed aggregates
	out <- we
}

// aggDBA receives partitioned weighted means from channel in and reduce them into a single mean for each cluster
// when finished send the result to out channel.
func aggDBA(sp core.Space, in <-chan []msg, out chan<- []msg) {
	var result []msg
	for we := range in {
		if result == nil {
			// first message, just take it as current result
			result = we
		} else {
			// combine subsequent messages to result
			for i := 0; i < len(result); i++ {
				switch {
				case result[i].card==0:
					// first time we see this cluster, just take the element
					result[i] = we[i]

				case we[i].card >0:
					// combine subsequent elements for this cluster
					sp.Combine(result[i].dba, result[i].card, we[i].dba, we[i].card)
					result[i].card += we[i].card
				}
			}
		}
	}

	// send aggregated result
	out <- result
	close(out)
}

// Iterate implements a parallel kmeans iteration.
// It starts as many worker routines as available CPU to execute aggAssign and fan out all data to them.
// It starts also one aggregator routine that executes aggDBA to handle workers results.
func (support ParKMeansSupport) Iterate(km algo.KMeans, clust algo.Clust) algo.Clust {
	// channels
	var degree = runtime.NumCPU()
	var in = make(chan core.Elemt, degree)
	var pipe = make(chan []msg, degree)
	var out = make(chan []msg, 1)
	var wg = &sync.WaitGroup{}

	// start workers
	wg.Add(degree)
	for i := 0; i < degree; i++ {
		go aggAssign(clust, km.Space, in, pipe, wg)
	}

	// start aggregator
	go aggDBA(km.Space, pipe, out)

	// fan out data to workers
	for i := 0; i < len(km.Data); i++ {
		in <- km.Data[i]
	}
	// close worker in channel, this will stop the workers when data is exhausted
	close(in)

	// wait all workers to shutdown
	wg.Wait()
	// close message exchange channel to stop the aggregator when messages are exhausted
	close(pipe)

	// read and build the result
	var we = <-out

	var result = make(algo.Clust, len(we))
	for i := 0; i < len(clust); i++ {
		if we[i].card > 0 {
			result[i] = we[i].dba
		} else {
			result[i] = clust[i]
		}
	}

	return result
}

// NewKMeans create a new parallel kmeans algorithm.
func NewKMeans(conf algo.KMeansConf, initializer algo.Initializer) algo.KMeans {
	var km = algo.NewKMeans(conf, initializer)
	km.KMeansSupport = ParKMeansSupport{space: conf.Space}
	return km
}
