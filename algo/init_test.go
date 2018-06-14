package algo

import (
	"testing"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
)

func TestWeightedChoice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++ {
		sum[WeightedChoice(w, src)] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000 {
		t.Errorf("Value out of range")
	}
}

func TestDBA(t *testing.T) {
	var sp = core.RealSpace{}
	var elemts = []core.Elemt{[]float64{2.}, []float64{4.}}

	var dba = DBA(elemts, sp)
	if e := dba.([]float64)[0]; e != 3. {
		t.Error("Expected 3 got", e)
	}

	elemts = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.}, []float64{6.},
		[]float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{3.},
		[]float64{1.}, []float64{-1.}, []float64{4.}}

	var s = 0.
	for i := 0; i < len(elemts); i++ {
		s += (elemts[i]).([]float64)[0]
	}
	dba = DBA(elemts, sp)

	var m = s / float64(len(elemts))
	var e = dba.([]float64)[0]
	if m != e {
		t.Error("Expected", m, "got", e)
	}

	var elemts2 = make([]core.Elemt, len(elemts))
	for i := range elemts {
		e := (elemts[i]).([]float64)[0]
		elemts2[i] = []float64{e, 2*e}
	}

	var dba2 = DBA(elemts2, sp)
	var ee = dba2.([]float64)
	if ee[0] != m || ee[1] != 2*m {
		t.Error("Expected [m m] got", ee)
	}
}
