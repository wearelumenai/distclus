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
func CreateOC(name string, conf core.Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) (oc core.OnlineClust, err error) {
	switch strings.ToLower(name) {
	case "mcmc":
		var algo = mcmc.NewAlgo(conf, space, data, initializer, args...)
		oc = core.OnlineClust(&algo)
	case "kmeans":
		var algo = kmeans.NewAlgo(conf, space, data, initializer, args...)
		oc = core.OnlineClust(&algo)
	default:
		err = errors.New("Unknown algorithm. MCMC and KMEANS expected")
	}

	return
}
