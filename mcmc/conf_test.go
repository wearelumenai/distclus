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
