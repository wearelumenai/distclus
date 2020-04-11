package core

type dbaPartition struct {
	dbas  Clust
	cards []int
}

func parReduceDBA(centroids Clust, data []Elemt, space Space, degree int) (Clust, []int) {
	var parts = make([]dbaPartition, degree)

	var process = func(start int, end int, rank int) {
		dbaReduce(space, centroids, data[start:end], &parts[rank])
	}

	Par(process, len(data), degree)

	var aggr = dbaAggregate(parts, space)
	return buildResult(centroids, aggr)
}

func parDBAForLabels(centroids Clust, data []Elemt, labels []int, space Space, degree int) ([]Elemt, []int) {
	var parts = make([]dbaPartition, degree)

	var process = func(start int, end int, rank int) {
		dbaReduceForLabels(space, centroids, data[start:end], labels[start:end], &parts[rank])
	}

	Par(process, len(data), degree)

	var aggr = dbaAggregate(parts, space)

	return aggr.dbas, aggr.cards
}

func dbaReduceForLabels(space Space, centroids Clust, elemts []Elemt, labels []int, part *dbaPartition) {
	part.dbas, part.cards = centroids.ReduceDBAForLabels(elemts, labels, space)
}

func dbaReduce(space Space, centroids Clust, elemts []Elemt, part *dbaPartition) {
	part.dbas, part.cards = centroids.ReduceDBA(elemts, space)
}

func dbaAggregate(parts []dbaPartition, space Space) dbaPartition {
	var aggregate dbaPartition
	for _, other := range parts {
		if aggregate.dbas == nil {
			aggregate.dbas = other.dbas
			aggregate.cards = other.cards
		} else {
			aggregate = dbaCombine(space, aggregate, other)
		}
	}

	return aggregate
}

func dbaCombine(space Space, aggregate dbaPartition, other dbaPartition) dbaPartition {
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

func buildResult(data Clust, aggr dbaPartition) (Clust, []int) {
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
