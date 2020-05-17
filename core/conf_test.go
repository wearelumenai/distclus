package core_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
)

func TestK_ConfErrorDataPerIter(t *testing.T) {
	var conf = core.CtrlConf{DataPerIter: -10}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func Test_ConfErrorIterPerData(t *testing.T) {
	var conf = core.CtrlConf{IterPerData: -10}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func Test_ConfErrorIterFreq(t *testing.T) {
	var conf = core.CtrlConf{IterFreq: -10}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func Test_ConfErrorTimeout(t *testing.T) {
	var conf = core.CtrlConf{Timeout: -10}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func Test_ConfErrorIter(t *testing.T) {
	var conf = core.CtrlConf{Iter: -10}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}
