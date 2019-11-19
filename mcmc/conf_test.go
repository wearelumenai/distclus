package mcmc_test

import (
	"distclus/internal/test"
	"distclus/mcmc"
	"testing"
)

func TestMCMC_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.Iter = -10
	conf.Verify()
}

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
