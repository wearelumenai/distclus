package core

import "github.com/wearelumenai/distclus/figures"

// OCModel online clustering model
type OCModel interface {
	Centroids() Clust                       // clustering result
	Conf() Conf                             // algo conf
	Impl() Impl                             // algo impl
	Space() Space                           // data space
	Status() OCStatus                       // algo status
	RuntimeFigures() figures.RuntimeFigures // clustering figures
}

// Centroids Get the centroids currently found by the algorithm
func (algo *Algo) Centroids() (centroids Clust) {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.centroids
}

// Space returns space
func (algo *Algo) Space() Space {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.space
}

// RuntimeFigures returns specific algo properties
func (algo *Algo) RuntimeFigures() (figures figures.RuntimeFigures) {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.runtimeFigures
}

// Conf returns configuration
func (algo *Algo) Conf() Conf {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.conf
}

// Impl returns impl
func (algo *Algo) Impl() Impl {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.impl
}

// Status returns algorithm status and failed error
func (algo *Algo) Status() OCStatus {
	algo.model.RLock()
	defer algo.model.RUnlock()
	return algo.status
}

// SimpleOCModel simple for fast run execution
type SimpleOCModel struct {
	conf           Conf
	space          Space
	status         OCStatus
	runtimeFigures figures.RuntimeFigures
	centroids      Clust
	impl           Impl
}

func (model SimpleOCModel) Centroids() Clust {
	return model.centroids
}

func (model SimpleOCModel) Space() Space {
	return model.space
}

func (model SimpleOCModel) Status() OCStatus {
	return model.status
}
func (model SimpleOCModel) RuntimeFigures() figures.RuntimeFigures {
	return model.runtimeFigures
}
func (model SimpleOCModel) Conf() Conf {
	return model.conf
}
func (model SimpleOCModel) Impl() Impl {
	return model.impl
}

func NewSimpleOCModel(conf Conf, space Space, status OCStatus, runtimeFigures figures.RuntimeFigures, centroids Clust) OCModel {
	return SimpleOCModel{
		conf:           conf,
		space:          space,
		status:         status,
		runtimeFigures: runtimeFigures,
		centroids:      centroids,
		impl:           nil,
	}
}
