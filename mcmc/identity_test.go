package mcmc_test

import (
	"distclus/mcmc"
	"reflect"
	"testing"
)

func Test_IdentitySample(t *testing.T) {
	var id = mcmc.NewIdentity()
	var mu = [][]float64{{1.}, {2.}}
	var x = id.Sample(mu, 321)
	if !reflect.DeepEqual(x, mu) {
		t.Error("identity error")
	}
}

func Test_IdentityPdf(t *testing.T) {
	var id = mcmc.NewIdentity()
	var mu = [][]float64{{1.}, {2.}}
	var p1 = id.Pdf(mu, mu, 325)
	if p1 != 1 {
		t.Error("identity pdf error")
	}
	var p0 = id.Pdf([][]float64{{2.}, {2.}}, mu, 412)
	if p0 != 0 {
		t.Error("identity pdf error")
	}
}
