package kmeans_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/kmeans"
)

func TestKMeans_ConfErrorK(t *testing.T) {
	var conf = kmeans.Conf{K: -12, CtrlConf: core.CtrlConf{Iter: 10}}
	var err = conf.Verify()
	if err == nil {
		t.Error("error expected")
	}
}

func TestKMeans_ConfErrorNumCPU(t *testing.T) {
	var conf = kmeans.Conf{K: 1}
	conf.Verify()
	if conf.NumCPU == 0 {
		t.Error("0 CPU. Positive expected")
	}
}
