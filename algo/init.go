package algo

import (
	"golang.org/x/exp/rand"
	"distclus/core"
)

type Initializer = func(k int, nodes []core.Elemt, space core.Space, src *rand.Rand) Clust

// Run au kmeans++ on a batch to return a k centers configuration
func KmeansPPInitializer(k int, elemts []core.Elemt, space core.Space, src *rand.Rand) Clust {
	if k < 1 {
		panic("k is lower than 1")
	}

	if len(elemts) < k {
		panic("not enough elements to initialize")
	}

	var idx = src.Intn(len(elemts))
	var clust = Clust{elemts[idx]}

	for i := 1; i < k; i++ {
		clust = KmeansPPIter(clust, elemts, space, src)
	}

	return clust
}

// Run au kmeans++ iteration on a batch to return a k+1 centers configuration
func KmeansPPIter(clust Clust, batch []core.Elemt, space core.Space, src *rand.Rand) Clust {
	var dists = make([]float64, len(batch))

	for i, elt := range batch {
		var center, _ = clust.UAssign(elt, space)
		dists[i] = space.Dist(elt, center)
	}

	return append(clust, batch[WeightedChoice(dists, src)])
}

// Random clustering initializer
func RandInitializer(k int, elemts []core.Elemt, _ core.Space, src *rand.Rand) Clust {
	if len(elemts) < k {
		panic("not enough elements to initialize")
	}

	var clust = make(Clust, k)
	var choices = make([]int, k)
	var i int

	for i < k {
		var choice = src.Intn(len(elemts))
		var find = false

		for _, v := range choices {
			if v == choice {
				find = true
			}
		}

		if !find {
			clust[i] = elemts[choice]
			choices[i] = choice
			i++
		}
	}
	return clust
}

// Return index of random weighted choice
func WeightedChoice(weights []float64, rand *rand.Rand) (idx int) {
	var sum float64
	for _, x := range weights{
		sum += x
	}
	var cursor = rand.Float64() * sum
	for cursor > 0 {
		cursor -= weights[idx]
		idx++
	}
	return idx - 1
}
