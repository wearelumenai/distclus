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
