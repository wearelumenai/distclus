package kmeans_test

import (
	"testing"

	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/internal/test"
	"github.com/wearelumenai/distclus/v0/kmeans"
)

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.Conf{K: -12, Conf: core.Conf{Iter: 10}}
	conf.Verify()
}

func TestKMeans_ConfErrorNumCPU(t *testing.T) {
	var conf = kmeans.Conf{K: 1}
	conf.Verify()
	if conf.NumCPU == 0 {
		t.Error("0 CPU. Positive expected")
	}
}
