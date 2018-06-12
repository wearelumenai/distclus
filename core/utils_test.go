package core

import (
	"testing"
	"time"
	"golang.org/x/exp/rand"
)

func TestWeightedChoice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++{
		sum[WeightedChoice(w, src)] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000{
		t.Errorf("Value out of range")
	}
}