package kmeans

import (
	"golang.org/x/exp/rand"
	"distclus/core"
)

// Checks if clustering initialization is possible.
func check(k int, elemts []core.Elemt) bool {
	if k < 1 {
		panic("K is lower than 1")
	}

	if len(elemts) < k {
		return false
	}

	return true
}

// GivenInitializer initializes a clustering algorithm with the k first testPoints.
func GivenInitializer (k int, elemts []core.Elemt, space core.Space, _ *rand.Rand) (core.Clust, bool) {
	var ok = check(k, elemts)
	var clust= make(core.Clust, k)

	if ok {
		for i := 0; i < k; i++ {
			clust[i] = space.Copy(elemts[i])
		}
	}

	return clust, ok
}

// KMeansPPInitializer initializes a clustering algorithm with kmeans++
func KMeansPPInitializer(k int, elemts []core.Elemt, space core.Space, src *rand.Rand) (core.Clust, bool) {
	var ok = check(k, elemts)
	var clust = make(core.Clust, k)

	if ok {
		var draw= src.Intn(len(elemts))
		clust[0] = elemts[draw]

		for i := 1; i < k; i++ {
			clust[i] = KMeansPPIter(clust[:i], elemts, space, src)
		}
	}

	return clust, ok
}

// Run au kmeans++ iteration : draw an element the does not belong to clust
func KMeansPPIter(clust core.Clust, elemts []core.Elemt, space core.Space, src *rand.Rand) core.Elemt {
	var dists = make([]float64, len(elemts))

	for i, elt := range elemts {
		var _, _, dist = clust.Assign(elt, space)
		dists[i] = dist
	}

	var draw = WeightedChoice(dists, src)
	return space.Copy(elemts[draw])
}

// RandomInitializer initializes a clustering with random testPoints
func RandInitializer(k int, elemts []core.Elemt, space core.Space, src *rand.Rand) (core.Clust, bool) {
	var ok = check(k, elemts)
	var clust = make(core.Clust, k)

	if ok {
		var chosen= make(map[int]int)
		var i int

		for i < k {
			var choice= src.Intn(len(elemts))
			var _, found= chosen[choice]

			if !found {
				clust[i] = space.Copy(elemts[choice])
				chosen[choice] = i
				i++
			}
		}
	}

	return clust, ok
}

// Return random index given corresponding weights
func WeightedChoice(weights []float64, rand *rand.Rand) int {
	var sum float64
	var idx = 0

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
