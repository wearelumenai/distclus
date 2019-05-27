package core

type partitionLosses struct {
	losses []float64
	cards  []int
}

func parLossForLabels(centroids Clust, data []Elemt, labels []int, space Space, norm float64, degree int) ([]float64, []int) {
	var parts = make([]partitionLosses, degree)

	var process = func(start int, end int, rank int) {
		lossReduceForLabels(centroids, data[start:end], labels[start:end], space, norm, &parts[rank])
	}

	Par(process, len(data), degree)

	var aggr = lossAggregate(parts)
	return aggr.losses, aggr.cards
}

func parLoss(centroids Clust, data []Elemt, space Space, norm float64, degree int) ([]float64, []int) {
	var parts = make([]partitionLosses, degree)

	var process = func(start int, end int, rank int) {
		lossReduce(centroids, data[start:end], space, norm, &parts[rank])
	}

	Par(process, len(data), degree)

	var aggr = lossAggregate(parts)
	return aggr.losses, aggr.cards
}

func lossReduceForLabels(centroids Clust, elemts []Elemt, labels []int, space Space, norm float64,
	part *partitionLosses) {
	part.losses, part.cards = centroids.ReduceLossForLabels(elemts, labels, space, norm)
}

func lossReduce(centroids Clust, elemts []Elemt, space Space, norm float64,
	part *partitionLosses) {
	part.losses, part.cards = centroids.ReduceLoss(elemts, space, norm)
}

func lossAggregate(parts []partitionLosses) partitionLosses {
	var aggregate partitionLosses
	for _, agg := range parts {
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
