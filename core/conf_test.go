package core_test

import (
	"testing"

	"go.lumenai.fr/distclus/v0/core"
	"go.lumenai.fr/distclus/v0/internal/test"
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
