package clustering_go

import (
	"math/rand"
)

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
