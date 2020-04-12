package mcmc_test

import (
	"testing"

	"go.lumenai.fr/distclus/v0/internal/test"
	"go.lumenai.fr/distclus/v0/mcmc"
)

func TestMCMC_ConfErrorMaxK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 30
	conf.MaxK = 10
	conf.Verify()
}

func TestMCMC_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 0
	conf.Verify()
}

func Test_Defaults(t *testing.T) {
	var conf = mcmc.Conf{}
	conf.SetConfigDefaults()

	test.AssertFalse(t, conf.RGen == nil)
	test.AssertFalse(t, conf.ProbaK == nil)
	test.AssertTrue(t, conf.Norm == 2)
	test.AssertTrue(t, conf.MaxK == 16)
	test.AssertTrue(t, conf.B == 1)
}

func TestKMeans_ConfErrorNumCPU(t *testing.T) {
	var conf = mcmcConf
	conf.Verify()
	if conf.NumCPU == 0 {
		t.Error("0 CPU. Positive expected")
	}
}
