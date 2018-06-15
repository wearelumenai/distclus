package algo

import (
	"testing"
	"distclus/core"
	"math"
	"golang.org/x/exp/rand"
)

func TestMultivT_Pdf(t *testing.T) {
	var conf = MCMCConf{
		Dim:      3, FrameSize: 8, B: 100, Amp: 1,
		Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
		InitIter: 1, Space: core.RealSpace{},
	}

	var x = []core.Elemt{
		[]float64{1., 3.4, 5.4},
		[]float64{10., 9.2, 12.3},
		[]float64{-4.3, -1.2, -3.},
	}

	var mu = []core.Elemt{
		[]float64{1.2, 3.1, 5.8},
		[]float64{9.7, 9.8, 11.6},
		[]float64{-4.4, -1.9, -2.3},
	}

	var distrib = NewMultivT(MultivTConf{conf})
	var d0 = distrib.Pdf(mu[0], x[0])

	if math.Abs(d0-0.319520) > 1e-6 {
		t.Error("Expected 0.319520 got", d0)
	}

	var d1 = distrib.Pdf(mu[1], x[1])

	if math.Abs(d1-0.0286968) > 1e-6 {
		t.Error("Expected 0.0286968 got", d1)
	}

	var d2 = distrib.Pdf(mu[2], x[2])

	if math.Abs(d2-0.0253301) > 1e-6 {
		t.Error("Expected 0.0253301 got", d0)
	}
}

func TestMultivT_Sample(t *testing.T) {
	var conf = MCMCConf{
		Dim:      3, FrameSize: 8, B: 100, Amp: 1,
		Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
		InitIter: 1, Space: core.RealSpace{},
		RGen: rand.New(rand.NewSource(6305689164243)),
	}

	var distrib = NewMultivT(MultivTConf{conf})

	// var mu = []float64{1.2, 3.1, 5.8}
	var mu = []float64{0., 0., 0.}
	var m = make([]float64, 3)

	n := 100000
	for i := 0; i < n; i++ {
		var s = distrib.Sample(mu).([]float64)
		for j := 0; j < len(m); j++ {
			m[j] += s[j] / float64(n)
		}
	}

	for j := 0; j < len(m); j++ {
		if math.Abs(m[j]-mu[j]) > 2e-3 {
			t.Error("Expected", mu[j], "got", m[j])
		}
	}

}
