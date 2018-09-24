package kmeans_test

import (
	"distclus/kmeans"
	"distclus/real"
	"distclus/internal/test"
	"testing"
)

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.KMeansConf{Iter: -10, K: 3, Space: real.RealSpace{}}
	conf.Verify()
}

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.KMeansConf{Iter: 10, K: -3, Space: real.RealSpace{}}
	conf.Verify()
}