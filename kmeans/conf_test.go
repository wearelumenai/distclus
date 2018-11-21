package kmeans_test

import (
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/real"
	"testing"
)

var algoConf = core.AlgorithmConf{
	Space: real.RealSpace{},
}

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: -10}
	kmeans.Verify(conf)
}

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var aconf = algoConf
	var conf = kmeans.Conf{AlgorithmConf: aconf, K: -12, Iter: 10}
	kmeans.Verify(conf)
}
