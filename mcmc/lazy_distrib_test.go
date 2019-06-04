package mcmc_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/mcmc"
	"testing"
)

var lazySpace = euclid.NewSpace(euclid.Conf{})

func initializer(elemt core.Elemt) mcmc.Distrib {
	var tConf = mcmc.MultivTConf{
		Conf: mcmcConf,
		Dim:  lazySpace.Dim([]core.Elemt{elemt}),
		Nu:   3,
	}
	return mcmc.NewMultivT(tConf)
}

func TestLazyDistrib_New(t *testing.T) {
	var distrib = mcmc.NewLazyDistrib(initializer)
	var _, ok = interface{}(distrib).(mcmc.Distrib)
	if !ok {
		t.Error("mcmc.Distrib not implemented")
	}
}

func TestLazyDistrib_Sample(t *testing.T) {
	var distrib = mcmc.NewLazyDistrib(initializer)
	var x = distrib.Sample([]float64{1.2, 1.1}, 3)
	if len(x.([]float64)) != 2 {
		t.Error("sample should have dimension 2")
	}
}

func TestLazyDistrib_Pdf(t *testing.T) {
	var distrib = mcmc.NewLazyDistrib(initializer)
	var mu = []float64{1.2, 1.1}
	var x = distrib.Sample(mu, 3)
	var p = distrib.Pdf(x, mu, 3)
	var p0 = initializer(x).Pdf(x, mu, 3)
	if p != p0 {
		t.Error("expected", p0, "got", p)
	}
}
