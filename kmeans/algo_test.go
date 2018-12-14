package kmeans_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/real"
	"testing"
)

var conf = core.Conf{ImplConf: kmeans.Conf{K: 1}, SpaceConf: nil}
var data = []core.Elemt{}
var initializer = kmeans.GivenInitializer

func Test_NewSeqAlgo(t *testing.T) {
	kmeans.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
	)
}

func Test_NewParAlgo(t *testing.T) {
	kmeansConf := conf.ImplConf.(kmeans.Conf)
	kmeansConf.Par = true
	kmeans.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
	)
}
