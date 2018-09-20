package kmeans

import (
	"distclus/core"
	"sync"
	"runtime"
)

func NewParKMeans(conf KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	var km = NewSeqKMeans(conf, initializer, data)
	var support = ParKMeansSupport{}
	support.buffer = &km.Buffer
	support.space = km.Space
	support.degree = runtime.NumCPU()
	km.KMeansSupport = &support
	return km
}

type ParKMeansSupport struct {
	buffer *core.Buffer
	space core.Space
	degree int
}

type workerSupport struct {
	ParKMeansSupport
	out chan msgKMeans
	wg *sync.WaitGroup
}

type msgKMeans struct {
	dbas core.Clust
	cards []int
}

func (support *ParKMeansSupport) Iterate(clust core.Clust) core.Clust {
	var workers = support.startKMeansWorkers(clust)
	var aggr = workers.assignAggregate()
	return support.buildResult(clust, aggr)
}

func (support *ParKMeansSupport) startKMeansWorkers(clust core.Clust) workerSupport {
	var offset = (len(support.buffer.Data)-1)/support.degree + 1
	var workers = workerSupport{	}
	workers.ParKMeansSupport = *support
	workers.out = make(chan msgKMeans, support.degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(support.degree)

	for i := 0; i < support.degree; i++ {
		var part = getChunk(i, offset, support.buffer.Data)
		go workers.assignMapReduce(clust, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return workers
}

func (support *workerSupport) assignMapReduce(clust core.Clust, elemts []core.Elemt) {
	defer support.wg.Done()

	var reduced msgKMeans
	reduced.dbas, reduced.cards = clust.AssignDBA(elemts, support.space)

	support.out <- reduced
}

func (support *workerSupport) assignAggregate() msgKMeans {
	var aggregate msgKMeans
	for other := range support.out {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = support.assignCombine(aggregate, other)
		}
	}

	return aggregate
}

func (support *workerSupport) assignCombine(aggregate msgKMeans, other msgKMeans) msgKMeans {
	for i := 0; i < len(aggregate.dbas); i++ {
		switch {
		case aggregate.cards[i] == 0:
			aggregate.dbas[i] = other.dbas[i]
			aggregate.cards[i] = other.cards[i]

		case other.cards[i] > 0:
			support.space.Combine(aggregate.dbas[i], aggregate.cards[i],
				other.dbas[i], other.cards[i])
			aggregate.cards[i] += other.cards[i]
		}
	}

	return aggregate
}

func (support *ParKMeansSupport)buildResult(clust core.Clust, aggr msgKMeans) core.Clust {
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

func getChunk(i int, offset int, elemts []core.Elemt) []core.Elemt {
	var start = i * offset
	var end = start + offset

	if end > len(elemts) {
		end = len(elemts)
	}

	return elemts[start:end]
}

