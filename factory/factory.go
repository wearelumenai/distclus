package factory

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/series"
	"distclus/vectors"
	"strings"
)

// CreateSpace create a new space
func CreateSpace(name string, conf core.SpaceConf) (space core.Space) {
	switch strings.ToLower(name) {
	case "vectors":
		space = vectors.NewSpace(conf)
	case "series":
		space = series.NewSpace(conf)
	}

	return
}

// CreateOC returns an algorithm by name and configuration
func CreateOC(name string, space string, conf core.Conf, data []core.Elemt, initializer core.Initializer, args ...interface{}) (oc core.OnlineClust) {
	var algo interface{}
	var finalSpace = CreateSpace(space, conf.SpaceConf)
	switch strings.ToLower(name) {
	case "mcmc":
		algo = mcmc.NewAlgo(conf, finalSpace, data, initializer, args...)
	case "kmeans":
		algo = kmeans.NewAlgo(conf, finalSpace, data, initializer, args...)
	}
	if algo != nil {
		oc = algo.(core.OnlineClust)
	}

	return
}
