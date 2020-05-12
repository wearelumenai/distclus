package core_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
)

func TestKMeans_ConfErrorDataPerIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.CtrlConf{DataPerIter: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorIterPerData(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.CtrlConf{IterPerData: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorIterFreq(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.CtrlConf{IterFreq: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorTimeout(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.CtrlConf{Timeout: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = core.CtrlConf{Iter: -10}
	conf.Verify()
}
