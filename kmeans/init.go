package kmeans

import (
	"distclus/core"
	"errors"
	"strings"

	"golang.org/x/exp/rand"
)

var initializersByNames = map[string]core.Initializer{
	"given": GivenInitializer,
	"pp":    PPInitializer,
	"rand":  RandInitializer,
}

// CreateInitializer creates an initializer with a name
func CreateInitializer(name string) core.Initializer {
	return initializersByNames[strings.ToLower(name)]
}

// Checks if clustering initialization is possible.
func check(k int, elemts []core.Elemt) (err error) {
	if k < 1 {
		err = errors.New("K is lower than 1")
	} else if len(elemts) < k {
		err = errors.New("Less elements than k")
	}

	return
}

// GivenInitializer initializes a clustering algorithm with the k first testPoints.
func GivenInitializer(k int, elemts []core.Elemt, space core.Space, _ *rand.Rand) (centroids core.Clust, err error) {
	err = check(k, elemts)
	centroids = make(core.Clust, k)

	if err == nil {
		for i := 0; i < k; i++ {
			centroids[i] = space.Copy(elemts[i])
		}
	}

	return
}

// PPInitializer initializes a clustering algorithm with kmeans++
func PPInitializer(k int, elemts []core.Elemt, space core.Space, src *rand.Rand) (centroids core.Clust, err error) {
	err = check(k, elemts)
	centroids = make(core.Clust, k)

	if err == nil {
		var draw = src.Intn(len(elemts))
		centroids[0] = elemts[draw]

		for i := 1; i < k; i++ {
			centroids[i] = PPIter(centroids[:i], elemts, space, src)
		}
	}

	return
}

// PPIter runs a kmeans++ iteration : draw an element the does not belong to clust
func PPIter(clust core.Clust, elemts []core.Elemt, space core.Space, src *rand.Rand) core.Elemt {
	var dists = make([]float64, len(elemts))

	for i, elt := range elemts {
		var _, _, dist = clust.Assign(elt, space)
		dists[i] = dist
	}

	var draw = WeightedChoice(dists, src)
	return space.Copy(elemts[draw])
}

// RandInitializer initializes a clustering with random testPoints
func RandInitializer(k int, elemts []core.Elemt, space core.Space, src *rand.Rand) (centroids core.Clust, err error) {
	err = check(k, elemts)
	centroids = make(core.Clust, k)

	if err == nil {
		var chosen = make(map[int]int)
		var i int

		for i < k {
			var choice = src.Intn(len(elemts))
			var _, found = chosen[choice]

			if !found {
				centroids[i] = space.Copy(elemts[choice])
				chosen[choice] = i
				i++
			}
		}
	}

	return
}

// WeightedChoice returns random index given corresponding weights
func WeightedChoice(weights []float64, rand *rand.Rand) int {
	var sum float64
	var idx int

	for _, x := range weights {
		sum += x
	}

	var cursor = rand.Float64() * sum
	for cursor > 0 {
		cursor -= weights[idx]
		idx++
	}
	return idx - 1
}
