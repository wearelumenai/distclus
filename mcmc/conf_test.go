package mcmc_test

import (
	"distclus/internal/test"
	"testing"
)

func TestMCMC_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = mcmcConf
	conf.McmcIter = -10
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
