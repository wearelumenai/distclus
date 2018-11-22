package kmeans_test

import (
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"
)

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.Conf{K: 3, Iter: -10}
	kmeans.Verify(conf)
}

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.Conf{K: -12, Iter: 10}
	kmeans.Verify(conf)
}
