package kmeans_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/real"
	"distclus/internal/test"
	"golang.org/x/exp/rand"
	"testing"
)

var seed = uint64(187236548914256543)
var algoConf = core.AlgoConf{
	InitK: 3,
	Space: real.RealSpace{},
	RGen: rand.New(rand.NewSource(seed)),
}

func TestKMeans_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: -10}
	conf.Verify()
}

func TestKMeans_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var aconf = algoConf
	aconf.InitK = -12
	var conf = kmeans.KMeansConf{AlgoConf: aconf, Iter: 10}
	conf.Verify()
}