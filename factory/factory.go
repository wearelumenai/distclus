package factory

import (
	"distclus/complex"
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"distclus/series"
	"fmt"
	"reflect"
	"strings"
)

// CreateSpace create a new space
func CreateSpace(name string, conf core.Conf) (space core.Space, err error) {
	switch strings.ToLower(name) {
	case "real":
		space = real.NewSpace(conf)
	case "complex":
		space = complex.NewSpace(conf)
	case "series":
		space = series.NewSpace(conf)
	default:
		err = fmt.Errorf("Unknown space. real, series or complex expected")
	}

	return
}

// CreateAlgo returns an algorithm by name and configuration
func CreateAlgo(name string, space string, par bool, data []Elemt, conf Conf, initializer *Initializer) (algo *Algo, err error) {
	var implFunc func(conf Conf)
	switch strings.ToLower(name) {
	case "mcmc":
		if par {
			implFunc = mcmc.NewParImpl
		} else {
			implFunc = mcmc.NewSeqImpl
		}
	case "kmeans":
		if par {
			implFunc = kmeans.NewParImpl
		} else {
			implFunc = kmeans.NewSeqImpl
		}
	default:
		err = fmt.Errorf("Unknown algorithm. %v expected", reflect.ValueOf(implementationsByName).MapKeys())
	}
	if implFunc != nil {
		var algoSpace = CreateSpace(space, conf)
		var impl = implFunc(conf, data, initializer)
		algo = NewAlgo(conf, impl, algoSpace)
	}
	return
}
