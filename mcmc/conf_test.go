package mcmc_test

import (
	"testing"

	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/mcmc"
)

func TestMCMC_ConfErrorMaxK(t *testing.T) {
	var conf = mcmcConf
	conf.InitK = 30
	conf.MaxK = 10
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func TestMCMC_ConfErrorK(t *testing.T) {
	var conf = mcmcConf
	conf.InitK = 0
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func Test_Defaults(t *testing.T) {
	var conf = mcmc.Conf{}
	conf.SetDefaultValues()

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
