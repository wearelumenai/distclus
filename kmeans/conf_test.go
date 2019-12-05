package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"
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
