package mcmc_test

import (
	"distclus/internal/test"
	"distclus/mcmc"
	"testing"
)

func TestMCMC_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.McmcIter = -10
	mcmc.Verify(conf)
}

func TestMCMC_ConfErrorMaxK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 30
	conf.MaxK = 10
	mcmc.Verify(conf)
}

func TestMCMC_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 0
	mcmc.Verify(conf)
}

func Test_Defaults(t *testing.T) {
	var conf = mcmc.Conf{}
	mcmc.SetConfigDefaults(&conf)

	test.AssertFalse(t, conf.RGen == nil)
	test.AssertFalse(t, conf.ProbaK == nil)
	test.AssertTrue(t, conf.Norm == 2)
	test.AssertTrue(t, conf.MaxK == 16)
	test.AssertTrue(t, conf.B == 1)
	test.AssertTrue(t, conf.Nu == 3)
	test.AssertTrue(t, conf.Iter == 1)
}
