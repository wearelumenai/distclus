package kmeans

import (
	"distclus/core"
	"sync"
	"runtime"
)

func NewParKMeans(conf KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	var km = NewSeqKMeans(conf, initializer, data)
	var strategy = ParKMeansStrategy{}
	strategy.Buffer = &km.template.Buffer
	strategy.Config = km.config
	strategy.Degree = runtime.NumCPU()
	km.strategy = &strategy
	return km
}

type ParKMeansStrategy struct {
	Config KMeansConf
	Buffer *core.Buffer
	Degree int
}

type workerSupport struct {
	ParKMeansStrategy
	out chan msgKMeans
	wg *sync.WaitGroup
}

type msgKMeans struct {
	dbas core.Clust
	cards []int
}

func (strategy *ParKMeansStrategy) Iterate(clust core.Clust) core.Clust {
	var workers = strategy.startKMeansWorkers(clust)
	var aggr = workers.assignAggregate()
	return strategy.buildResult(clust, aggr)
}

func (strategy *ParKMeansStrategy) startKMeansWorkers(clust core.Clust) workerSupport {
	var offset = (len(strategy.Buffer.Data)-1)/strategy.Degree + 1
	var workers = workerSupport{	}
	workers.ParKMeansStrategy = *strategy
	workers.out = make(chan msgKMeans, strategy.Degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(strategy.Degree)

	for i := 0; i < strategy.Degree; i++ {
		var part = core.GetChunk(i, offset, strategy.Buffer.Data)
		go workers.assignMapReduce(clust, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return workers
}

func (strategy *workerSupport) assignMapReduce(clust core.Clust, elemts []core.Elemt) {
	defer strategy.wg.Done()

	var reduced msgKMeans
	reduced.dbas, reduced.cards = clust.AssignDBA(elemts, strategy.Config.Space)

	strategy.out <- reduced
}

func (strategy *workerSupport) assignAggregate() msgKMeans {
	var aggregate msgKMeans
	for other := range strategy.out {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = strategy.assignCombine(aggregate, other)
		}
	}

	return aggregate
}

func (strategy *workerSupport) assignCombine(aggregate msgKMeans, other msgKMeans) msgKMeans {
	for i := 0; i < len(aggregate.dbas); i++ {
		switch {
		case aggregate.cards[i] == 0:
			aggregate.dbas[i] = other.dbas[i]
			aggregate.cards[i] = other.cards[i]

		case other.cards[i] > 0:
			strategy.Config.Space.Combine(aggregate.dbas[i], aggregate.cards[i],
				other.dbas[i], other.cards[i])
			aggregate.cards[i] += other.cards[i]
		}
	}

	return aggregate
}

func (strategy *ParKMeansStrategy)buildResult(clust core.Clust, aggr msgKMeans) core.Clust {
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

