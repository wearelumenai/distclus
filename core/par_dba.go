package core

import "sync"

type dbaWorker struct {
	parts []partitionDBA
	wg    *sync.WaitGroup
}

type partitionDBA struct {
	dbas  Clust
	cards []int
}

func parAssignDBA(centroids Clust, data []Elemt, space Space, degree int) (Clust, []int) {
	var offset = (len(data)-1)/degree + 1
	var workers = dbaWorker{
		parts: make([]partitionDBA, degree),
		wg:    &sync.WaitGroup{},
	}
	workers.wg.Add(degree)

	for i := 0; i < degree; i++ {
		var part = GetChunk(i, offset, data)
		go workers.assignMapReduce(space, centroids, part, i)
	}

	workers.wg.Wait()

	var aggr = workers.assignAggregate(space)
	return buildResult(centroids, aggr)
}

func (strategy *dbaWorker) assignMapReduce(space Space, centroids Clust, elemts []Elemt, index int) {
	defer strategy.wg.Done()
	strategy.parts[index].dbas, strategy.parts[index].cards = centroids.AssignDBA(elemts, space)
}

func (strategy *dbaWorker) assignAggregate(space Space) partitionDBA {
	var aggregate partitionDBA
	for _, other := range strategy.parts {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = strategy.assignCombine(space, aggregate, other)
		}
	}

	return aggregate
}

func (strategy *dbaWorker) assignCombine(space Space, aggregate partitionDBA,
	other partitionDBA) partitionDBA {
	for i := 0; i < len(aggregate.dbas); i++ {
		switch {
		case aggregate.cards[i] == 0:
			aggregate.dbas[i] = other.dbas[i]
			aggregate.cards[i] = other.cards[i]

		case other.cards[i] > 0:
			aggregate.dbas[i] = space.Combine(
				aggregate.dbas[i], aggregate.cards[i],
				other.dbas[i], other.cards[i],
			)
			aggregate.cards[i] += other.cards[i]
		}
	}

	return aggregate
}

func buildResult(data Clust, aggr partitionDBA) (Clust, []int) {
	var result = make(Clust, len(aggr.dbas))
	for i := 0; i < len(data); i++ {
		if aggr.cards[i] > 0 {
			result[i] = aggr.dbas[i]
		} else {
			result[i] = data[i]
		}
	}
	return result, aggr.cards
}
