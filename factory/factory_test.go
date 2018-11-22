package factory_test

import (
	"distclus/core"
	"distclus/factory"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/series"

	"testing"
)

var ocs = map[string]core.Conf{
	"MCMC":    mcmc.Conf{},
	"mcmc":    mcmc.Conf{},
	"kmeans":  kmeans.Conf{},
	"unknown": nil,
}

var spaces = map[string]core.Conf{
	"Real":    series.Conf{},
	"real":    series.Conf{},
	"complex": series.Conf{},
	"series":  series.Conf{},
	"unknown": nil,
}

func Test_CreateSpace(t *testing.T) {
	for name, conf := range spaces {
		var space, err = factory.CreateSpace(name, conf)
		if conf != nil {
			if space == nil {
				t.Error("space not created")
			}
			if err != nil {
				t.Error("an error has been thrown")
			}
		} else {
			if space != nil {
				t.Error("space has been created")
			}
			if err == nil {
				t.Error("an error has not been thrown")
			}
		}
	}
}

func Test_CreateOC(t *testing.T) {
	for oc, conf := range ocs {
		for space, spaceConf := range spaces {
			algoSpace, _ := factory.CreateSpace(space, spaceConf)
			if algoSpace != nil {
				for _, par := range []bool{true, false} {
					var _, err = factory.CreateOC(oc, conf, algoSpace, par, nil, nil)
					if conf != nil {
						if err != nil {
							t.Error("an error has been thrown")
						}
					} else {
						if err == nil {
							t.Error("an error has not been thrown")
						}
					}
				}
			}
		}
	}
}
