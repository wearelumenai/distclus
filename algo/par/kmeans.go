package par

import (
	"distclus/algo"
	"distclus/core"
	"sync"
	"runtime"
)

func NewKMeans(conf algo.KMeansConf, initializer core.Initializer, data []core.Elemt) algo.KMeans {
	var km = algo.NewKMeans(conf, initializer, data)
	km.KMeansSupport = ParKMeansSupport{space: conf.Space}
	return km
}

type ParKMeansSupport struct {
	space core.Space
}

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

type msgKMeans struct {
	dbas core.Clust
	cards []int
}

func assignMapReduce(clust core.Clust, elemts []core.Elemt, space core.Space, out chan<- msgKMeans, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced msgKMeans
	reduced.dbas, reduced.cards = clust.AssignDBA(elemts, space)

	out <- reduced
}

func assignAggregate(space core.Space, in <-chan msgKMeans) msgKMeans {
	var aggregate msgKMeans
	for other := range in {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = assignCombine(aggregate, other, space)
		}
	}

	return aggregate
}

func assignCombine(aggregate msgKMeans, other msgKMeans, space core.Space) msgKMeans {
	for i := 0; i < len(aggregate.dbas); i++ {
		switch {
		case aggregate.cards[i] == 0:
			aggregate.dbas[i] = other.dbas[i]
			aggregate.cards[i] = other.cards[i]

		case other.cards[i] > 0:
			space.Combine(aggregate.dbas[i], aggregate.cards[i],
				other.dbas[i], other.cards[i])
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

