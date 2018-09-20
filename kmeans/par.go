package kmeans

import (
	"distclus/core"
	"sync"
	"runtime"
)

func NewParKMeans(conf KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	var km = NewSeqKMeans(conf, initializer, data)
	km.KMeansSupport = ParKMeansSupport{buffer:&km.Buffer, space:km.Space}
	return km
}

type ParKMeansSupport struct {
	buffer *core.Buffer
	space core.Space
	degree int
	offset int
}

func (support ParKMeansSupport) Iterate(clust core.Clust) core.Clust {
	var out = support.startKMeansWorkers(clust)
	var aggr = support.assignAggregate(out)
	return support.buildResult(clust, aggr)
}

func (support ParKMeansSupport) startKMeansWorkers(clust core.Clust) (chan msgKMeans) {
	var degree = runtime.NumCPU()
	var offset = (len(support.buffer.Data)-1)/degree + 1
	var out = make(chan msgKMeans, degree)
	var wg = &sync.WaitGroup{}
	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var part = getChunk(i, offset, support.buffer.Data)
		go support.assignMapReduce(clust, part, out, wg)
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

type msgKMeans struct {
	dbas core.Clust
	cards []int
}

func (support ParKMeansSupport) assignMapReduce(clust core.Clust, elemts []core.Elemt, out chan<- msgKMeans, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced msgKMeans
	reduced.dbas, reduced.cards = clust.AssignDBA(elemts, support.space)

	out <- reduced
}

func (support ParKMeansSupport) assignAggregate(in <-chan msgKMeans) msgKMeans {
	var aggregate msgKMeans
	for other := range in {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = support.assignCombine(aggregate, other)
		}
	}

	return aggregate
}

func (support ParKMeansSupport) assignCombine(aggregate msgKMeans, other msgKMeans) msgKMeans {
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

func (ParKMeansSupport)buildResult(clust core.Clust, aggr msgKMeans) core.Clust {
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

