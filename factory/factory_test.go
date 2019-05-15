package factory_test

import (
	"distclus/core"
	"distclus/cosinus"
	"distclus/factory"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/series"
	"distclus/vectors"
	"golang.org/x/exp/rand"
	"reflect"

	"testing"
)

var mcmcConf = mcmc.Conf{
	InitK:     3,
	FrameSize: 8,
	RGen:      rand.New(rand.NewSource(6305689164243)),
	B:         100, Amp: 1,
	Norm: 2, Nu: 3, McmcIter: 20,
	InitIter: 1,
}

var ocs = []core.ImplConf{
	mcmcConf,
	kmeans.Conf{K: 1, Iter: 1},
}

var spaces = []core.SpaceConf{
	series.Conf{},
	vectors.Conf{},
	cosinus.Conf{},
}

func Test_CreateSpace(t *testing.T) {
	for _, conf := range spaces {
		var space = factory.CreateSpace(conf)
		if space == nil {
			t.Error("space has been created")
		}
	}
}

func getData(spaceConf core.SpaceConf) (data []core.Elemt) {
	data = make([]core.Elemt, 1)
	switch spaceConf.(type) {
	case vectors.Conf:
		data[0] = []float64{0.}
	case series.Conf:
		data[0] = [][]float64{{0.}}
	}
	return
}

func Test_CreateOC(t *testing.T) {
	for _, implConf := range ocs {
		for _, spaceConf := range spaces {
			algoSpace := factory.CreateSpace(spaceConf)
			if algoSpace != nil {
				data := getData(spaceConf)
				var algo, _ = factory.CreateOC(implConf, spaceConf, data, nil)
				if algo == nil {
					t.Error("algorithm should have been constructed")
				}
				if reflect.TypeOf((algo).(*core.Algo).Space()) != reflect.TypeOf(algoSpace) {
					t.Error("wrong space type")
				}
			}
		}
	}
}
