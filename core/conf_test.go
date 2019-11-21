package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"testing"
)

func TestKMeans_ConfErrorDataPerIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{DataPerIter: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorIterFreq(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{IterFreq: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorTimeout(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{Timeout: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorNumCPU(t *testing.T) {
	var conf = core.Conf{}
	conf.Verify()
	if conf.NumCPU == 0 {
		t.Error("0 CPU. Positive expected")
	}
}
