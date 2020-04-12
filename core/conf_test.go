package core_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
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
