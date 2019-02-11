package core

import "sync"

type lossWorker struct {
	out chan lossMessage
	wg  *sync.WaitGroup
}

type lossMessage struct {
	losses []float64
	cards  []int
}

func ParLosses(centroids Clust, data []Elemt, space Space, norm float64, degree int) ([]float64, []int) {
	var offset = (len(data)-1)/degree + 1
	var workers = lossWorker{}
	workers.out = make(chan lossMessage, degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(degree)

	for i := 0; i < degree; i++ {
		var part = GetChunk(i, offset, data)
		go workers.lossMapReduce(norm, space, centroids, part)
	}

	workers.wg.Wait()
	close(workers.out)

	var aggr = workers.lossAggregate()
	return aggr.losses, aggr.cards
}

func (strategy *lossWorker) lossMapReduce(norm float64, space Space, centroids Clust, elemts []Elemt) {
	defer strategy.wg.Done()

	var reduced lossMessage
	reduced.losses, reduced.cards = centroids.Losses(elemts, space, norm)

	strategy.out <- reduced
}

func (strategy *lossWorker) lossAggregate() lossMessage {
	var aggregate lossMessage
	for agg := range strategy.out {
		if aggregate.losses == nil {
			aggregate.losses = make([]float64, len(agg.losses))
			aggregate.cards = make([]int, len(agg.cards))
			copy(aggregate.losses, agg.losses)
			copy(aggregate.cards, agg.cards)
		} else {
			for i := range agg.losses {
				aggregate.losses[i] += agg.losses[i]
				aggregate.cards[i] += agg.cards[i]
			}
		}
	}
	return aggregate
}
