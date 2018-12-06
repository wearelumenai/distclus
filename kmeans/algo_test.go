package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
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

func Test_Reset(t *testing.T) {
	algo := kmeans.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
	)

	test.DoTestReset(t, &algo, kmeans.Conf{K: 1})
}
