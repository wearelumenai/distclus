package mcmc_test

import (
	"testing"

	"go.lumenai.fr/distclus/v0/core"
	"go.lumenai.fr/distclus/v0/euclid"
	"go.lumenai.fr/distclus/v0/mcmc"
)

var lateSpace = euclid.NewSpace()

func initializer(elemt core.Elemt) mcmc.Distrib {
	var tConf = mcmc.MultivTConf{
		Dim: lateSpace.Dim([]core.Elemt{elemt}),
		Nu:  3,
	}
	return mcmc.NewMultivT(tConf)
}

func TestLateDistrib_New(t *testing.T) {
	var distrib = mcmc.NewLateDistrib(initializer)
	var _, ok = interface{}(distrib).(mcmc.Distrib)
	if !ok {
		t.Error("mcmc.Distrib not implemented")
	}
}

func TestLateDistrib_Sample(t *testing.T) {
	var distrib = mcmc.NewLateDistrib(initializer)
	var x = distrib.Sample([]float64{1.2, 1.1}, 3)
	if len(x.([]float64)) != 2 {
		t.Error("sample should have dimension 2")
	}
}

func TestLateDistrib_Pdf(t *testing.T) {
	var distrib = mcmc.NewLateDistrib(initializer)
	var mu = []float64{1.2, 1.1}
	var x = distrib.Sample(mu, 3)
	var p = distrib.Pdf(x, mu, 3)
	var p0 = initializer(x).Pdf(x, mu, 3)
	if p != p0 {
		t.Error("expected", p0, "got", p)
	}
}
