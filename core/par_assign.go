package core

import "sync"

type assignWorker struct {
	out chan assignMessage
	wg  *sync.WaitGroup
}

type assignMessage struct {
	dbas  Clust
	cards []int
}

func parAssign(centroids Clust, data []Elemt, space Space, degree int) (Clust, []int) {
	var offset = (len(data)-1)/degree + 1
	var workers = assignWorker{
		out: make(chan assignMessage, degree),
		wg:  &sync.WaitGroup{},
	}
	workers.wg.Add(degree)

	for i := 0; i < degree; i++ {
		var part = GetChunk(i, offset, data)
		go workers.assignMapReduce(space, centroids, part)
	}

	workers.wg.Wait()
	close(workers.out)

	var aggr = workers.assignAggregate(space)
	return buildResult(centroids, aggr)
}

func (strategy *assignWorker) assignMapReduce(space Space, centroids Clust, elemts []Elemt) {
	defer strategy.wg.Done()

	var reduced assignMessage
	reduced.dbas, reduced.cards = centroids.AssignDBA(elemts, space)

	strategy.out <- reduced
}

func (strategy *assignWorker) assignAggregate(space Space) assignMessage {
	var aggregate assignMessage
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

func (strategy *assignWorker) assignCombine(space Space, aggregate assignMessage, other assignMessage) assignMessage {
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

func buildResult(data Clust, aggr assignMessage) (Clust, []int) {
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
