package core

type dbaParition struct {
	dbas  Clust
	cards []int
}

func parReduceDBA(centroids Clust, data []Elemt, space Space, degree int) (Clust, []int) {
	var parts = make([]dbaParition, degree)

	var process = func(part []Elemt, start int, end int, rank int) {
		dbaReduce(space, centroids, part, &parts[rank])
	}

	Par(process, data, degree)

	var aggr = dbaAggregate(parts, space)
	return buildResult(centroids, aggr)
}

func dbaReduce(space Space, centroids Clust, elemts []Elemt, part *dbaParition) {
	part.dbas, part.cards = centroids.ReduceDBA(elemts, space)
}

func dbaAggregate(parts []dbaParition, space Space) dbaParition {
	var aggregate dbaParition
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

func dbaCombine(space Space, aggregate dbaParition, other dbaParition) dbaParition {
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

func buildResult(data Clust, aggr dbaParition) (Clust, []int) {
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
