package mcmc_test

import (
	"lumenai.fr/v0/distclus/pkg/mcmc"
	"reflect"
	"testing"
)

func Test_IdentitySample(t *testing.T) {
	var id = mcmc.NewDirac()
	var mu = [][]float64{{1.}, {2.}}
	var x = id.Sample(mu, 321)
	if !reflect.DeepEqual(x, mu) {
		t.Error("identity error")
	}
}

func Test_IdentityPdf(t *testing.T) {
	var id = mcmc.NewDirac()
	var mu = [][]float64{{1.}, {2.}}
	var p1 = id.Pdf(mu, mu, 325)
	if p1 != 1 {
		t.Error("identity pdf error")
	}
}
