package factory

import (
	"distclus/core"
	"distclus/cosinus"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/series"
	"distclus/vectors"
)

// CreateSpace create a new space
func CreateSpace(spaceConf core.SpaceConf) (space core.Space) {
	switch conf := spaceConf.(type) {
	case vectors.Conf:
		space = vectors.NewSpace(conf)
	case series.Conf:
		space = series.NewSpace(conf)
	case cosinus.Conf:
		space = cosinus.NewSpace(conf)
	}
	return
}

// CreateOC returns an algorithm by name and configuration
func CreateOC(implConf core.ImplConf, spaceConf core.SpaceConf, data []core.Elemt, initializer core.Initializer,
	args ...interface{}) (oc core.OnlineClust, space core.Space) {
	space = CreateSpace(spaceConf)
	switch conf := implConf.(type) {
	case mcmc.Conf:
		oc = mcmc.NewAlgo(conf, space, data, initializer, args...)
	case kmeans.Conf:
		oc = kmeans.NewAlgo(conf, space, data, initializer, args...)
	}
	return
}
