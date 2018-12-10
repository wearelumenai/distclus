package factory

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"distclus/series"
	"strings"
)

// CreateSpace create a new space
func CreateSpace(name string, conf core.Conf) (space core.Space) {
	switch strings.ToLower(name) {
	case "real":
		space = real.NewSpace(conf)
	case "series":
		space = series.NewSpace(conf)
	}

	return
}

// CreateOC returns an algorithm by name and configuration
func CreateOC(name string, conf core.Conf, space core.Space, data []core.Elemt, initializer core.Initializer, args ...interface{}) (oc core.OnlineClust) {
	var algo interface{}
	switch strings.ToLower(name) {
	case "mcmc":
		algo = mcmc.NewAlgo(conf, space, data, initializer, args...)
	case "kmeans":
		algo = kmeans.NewAlgo(conf, space, data, initializer, args...)
	}
	if algo != nil {
		oc = algo.(core.OnlineClust)
	}

	return
}
