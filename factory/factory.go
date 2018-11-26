package factory

import (
	"distclus/complex"
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"distclus/series"
	"errors"
	"fmt"
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

// CreateOC returns an algorithm by name and configuration
func CreateOC(name string, conf core.Conf, space core.Space, par bool, data []core.Elemt, initializer core.Initializer, args ...interface{}) (algo core.OnlineClust, err error) {
	var impl core.Impl
	switch strings.ToLower(name) {
	case "mcmc":
		if par {
			simpl := mcmc.NewParImpl(conf, initializer, data, args...)
			impl = &simpl
		} else {
			simpl := mcmc.NewSeqImpl(conf, initializer, data, args...)
			impl = &simpl
		}
	case "kmeans":
		if par {
			simpl := kmeans.NewParImpl(conf, initializer, data, args...)
			impl = &simpl
		} else {
			simpl := kmeans.NewSeqImpl(conf, initializer, data, args...)
			impl = &simpl
		}
	default:
		err = errors.New("Unknown algorithm. MCMC and KMEANS expected")
	}
	if impl != nil {
		var newAlgo = core.NewAlgo(conf, impl, space)
		algo = &newAlgo
	}
	return
}
