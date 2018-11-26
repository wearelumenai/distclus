package kmeans

import (
	"distclus/core"
	"runtime"
	"sync"
)

// NewParImpl parallelizes algorithm implementation
func NewParImpl(conf core.Conf, initializer core.Initializer, data []core.Elemt, _ ...interface{}) (impl Impl) {
	impl = NewSeqImpl(conf, initializer, data)
	impl.strategy = ParStrategy{
		Degree: runtime.NumCPU(),
	}
	return
}

// ParStrategy parallelizes algorithm strategy
type ParStrategy struct {
	Degree int
}

type workerSupport struct {
	ParStrategy
	out chan msg
	wg  *sync.WaitGroup
}

type msg struct {
	dbas  core.Clust
	cards []int
}

// Iterate processes input cluster
func (strategy ParStrategy) Iterate(space core.Space, centroids core.Clust, buffer core.Buffer) core.Clust {
	var workers = strategy.startWorkers(space, centroids, buffer)
	var aggr = workers.assignAggregate(space)
	return strategy.buildResult(centroids, aggr)
}

func (strategy ParStrategy) startWorkers(space core.Space, centroids core.Clust, buffer core.Buffer) (workers workerSupport) {
	var offset = (len(buffer.Data())-1)/strategy.Degree + 1
	workers = workerSupport{
		ParStrategy: strategy,
		out:         make(chan msg, strategy.Degree),
		wg:          &sync.WaitGroup{},
	}
	workers.wg.Add(strategy.Degree)

	for i := 0; i < strategy.Degree; i++ {
		var part = core.GetChunk(i, offset, buffer.Data())
		go workers.assignMapReduce(space, centroids, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return
}

func (strategy *workerSupport) assignMapReduce(space core.Space, centroids core.Clust, elemts []core.Elemt) {
	defer strategy.wg.Done()

	var reduced msg
	reduced.dbas, reduced.cards = centroids.AssignDBA(elemts, space)

	strategy.out <- reduced
}

func (strategy *workerSupport) assignAggregate(space core.Space) msg {
	var aggregate msg
	for other := range strategy.out {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = strategy.assignCombine(space, aggregate, other)
		}
	}

	return aggregate
}

func (strategy *workerSupport) assignCombine(space core.Space, aggregate msg, other msg) msg {
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

func (strategy *ParStrategy) buildResult(data core.Clust, aggr msg) core.Clust {
	var result = make(core.Clust, len(aggr.dbas))
	for i := 0; i < len(data); i++ {
		if aggr.cards[i] > 0 {
			result[i] = aggr.dbas[i]
		} else {
			result[i] = data[i]
		}
	}
	return result
}
