package mcmc_test

import (
	"distclus/core"
	"distclus/mcmc"
	"math"
	"testing"

	"golang.org/x/exp/rand"
)

var mcmcConf = mcmc.Conf{
	InitK: 3,
	RGen:  rand.New(rand.NewSource(6305689164243)),
	B:     100,
	Amp:   1,
	Norm:  2,
	Conf:  core.Conf{FrameSize: 8},
}

var mvtConf = mcmc.MultivTConf{
	Dim: 3,
	Nu:  3,
}

func TestMultivT_Pdf(t *testing.T) {
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

	var distrib = mcmc.NewMultivT(mvtConf)
	var d0 = math.Exp(distrib.Pdf(mu[0], x[0], 8))

	if math.Abs(d0-0.319520) > 1e-6 {
		t.Error("Expected 0.319520 got", d0)
	}

	var d1 = math.Exp(distrib.Pdf(mu[1], x[1], 8))

	if math.Abs(d1-0.0286968) > 1e-6 {
		t.Error("Expected 0.0286968 got", d1)
	}

	var d2 = math.Exp(distrib.Pdf(mu[2], x[2], 8))

	if math.Abs(d2-0.0253301) > 1e-6 {
		t.Error("Expected 0.0253301 got", d0)
	}
}

func TestMultivT_Sample(t *testing.T) {
	var distrib = mcmc.NewMultivT(mvtConf)

	// var mu = []float64{1.2, 3.1, 5.8}
	var mu = []float64{0., 0., 0.}
	var m = make([]float64, 3)

	n := 1000000
	for i := 0; i < n; i++ {
		var s = distrib.Sample(mu, 8).([]float64)
		for j := 0; j < len(m); j++ {
			m[j] += s[j] / float64(n)
		}
	}

	for j := 0; j < len(m); j++ {
		if math.Abs(m[j]-mu[j]) > 1e-3 {
			t.Error("Expected", mu[j], "got", m[j])
		}
	}
}
