package kmeans_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/real"
	"testing"
)

var conf = kmeans.Conf{K: 1}
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
	conf.Par = true
	kmeans.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
	)
}
