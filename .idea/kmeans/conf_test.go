package kmeans_test

import (
	"distclus/core"
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
	var conf = kmeans.KMeansConf{AlgorithmConf: algoConf, K: 3, Iter: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var aconf = algoConf
	var conf = kmeans.KMeansConf{AlgorithmConf: aconf, K: -12, Iter: 10}
	conf.Verify()
}
