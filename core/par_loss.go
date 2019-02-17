package core

import "sync"

type lossWorker struct {
	parts []partitionLosses
	wg    *sync.WaitGroup
}

type partitionLosses struct {
	losses []float64
	cards  []int
}

func ParLosses(centroids Clust, data []Elemt, space Space, norm float64, degree int) ([]float64, []int) {
	var offset = (len(data)-1)/degree + 1
	var workers = lossWorker{
		parts: make([]partitionLosses, degree),
		wg:    &sync.WaitGroup{},
	}
	workers.wg.Add(degree)

	for i := 0; i < degree; i++ {
		var part = GetChunk(i, offset, data)
		go workers.lossMapReduce(centroids, part, space, norm, i)
	}

	workers.wg.Wait()

	var aggr = workers.lossAggregate()
	return aggr.losses, aggr.cards
}

func (strategy *lossWorker) lossMapReduce(centroids Clust, elemts []Elemt, space Space, norm float64, index int) {
	defer strategy.wg.Done()
	strategy.parts[index].losses, strategy.parts[index].cards = centroids.Losses(elemts, space, norm)
}

func (strategy *lossWorker) lossAggregate() partitionLosses {
	var aggregate partitionLosses
	for _, agg := range strategy.parts {
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
