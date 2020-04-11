package core_test

import (
	"lumenai.fr/v0/distclus/internal/test"
	"lumenai.fr/v0/distclus/pkg/core"
	"testing"
)

func TestKMeans_ConfErrorDataPerIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{DataPerIter: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorIterPerData(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{IterPerData: -10}
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

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.Conf{Iter: -10}
	conf.Verify()
}
