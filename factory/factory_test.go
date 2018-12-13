package factory_test

import (
	"distclus/core"
	"distclus/factory"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/series"
	"strings"

	"golang.org/x/exp/rand"

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

var ocs = map[string]core.ImplConf{
	"MCMC":    mcmcConf,
	"mcmc":    mcmcConf,
	"kmeans":  kmeans.Conf{K: 1, Iter: 1},
	"unknown": nil,
}

var spaces = map[string]core.SpaceConf{
	"Real":    series.Conf{},
	"real":    series.Conf{},
	"series":  series.Conf{},
	"unknown": nil,
}

func Test_CreateSpace(t *testing.T) {
	for name, conf := range spaces {
		var space = factory.CreateSpace(name, conf)
		if conf != nil {
			if space == nil {
				t.Error("space not created")
			}
		} else {
			if space != nil {
				t.Error("space has been created")
			}
		}
	}
}

func getData(space string) (data []core.Elemt) {
	data = make([]core.Elemt, 1)
	switch strings.ToLower(space) {
	case "real":
		data[0] = []float64{0.}
	case "series":
		data[0] = [][]float64{{0.}}
	}
	return
}

func Test_CreateOC(t *testing.T) {
	for oc, implConf := range ocs {
		for space, spaceConf := range spaces {
			algoSpace := factory.CreateSpace(space, spaceConf)
			if algoSpace != nil {
				data := getData(space)
				conf := core.Conf{ImplConf: implConf, SpaceConf: spaceConf}
				var algo = factory.CreateOC(oc, space, conf, data, nil)
				if implConf != nil {
					if algo == nil {
						t.Error("an error has been thrown")
					}
				} else {
					if algo != nil {
						t.Error("an error has not been thrown")
					}
				}
			}
		}
	}
}
