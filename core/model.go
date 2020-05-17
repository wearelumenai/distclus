package core

// OCModel online clustering model
type OCModel interface {
	Centroids() Clust               // clustering result
	Conf() Conf                     // algo conf
	Impl() Impl                     // algo impl
	Space() Space                   // data space
	Status() OCStatus               // algo status
	RuntimeFigures() RuntimeFigures // clustering figures
}

// Centroids Get the centroids currently found by the algorithm
func (algo *Algo) Centroids() (centroids Clust) {
	algo.modelMutex.RLock()
	defer algo.modelMutex.RUnlock()
	return algo.centroids
}

// Space returns space
func (algo *Algo) Space() Space {
	algo.modelMutex.RLock()
	defer algo.modelMutex.RUnlock()
	return algo.space
}

// RuntimeFigures returns specific algo properties
func (algo *Algo) RuntimeFigures() (figures RuntimeFigures) {
	algo.modelMutex.RLock()
	defer algo.modelMutex.RUnlock()
	return algo.runtimeFigures
}

// Conf returns configuration
func (algo *Algo) Conf() Conf {
	algo.modelMutex.RLock()
	defer algo.modelMutex.RUnlock()
	return algo.conf
}

// Impl returns impl
func (algo *Algo) Impl() Impl {
	algo.modelMutex.RLock()
	defer algo.modelMutex.RUnlock()
	return algo.impl
}

// Status returns algorithm status and failed error
func (algo *Algo) Status() OCStatus {
	algo.statusMutex.RLock()
	defer algo.statusMutex.RUnlock()
	return algo.status
}

// SimpleOCModel simple for fast run execution
type SimpleOCModel struct {
	conf           Conf
	space          Space
	status         OCStatus
	runtimeFigures RuntimeFigures
	centroids      Clust
	impl           Impl
}

// Centroids for simpleocmodel
func (model SimpleOCModel) Centroids() Clust {
	return model.centroids
}

// Space for simpleocmodel
func (model SimpleOCModel) Space() Space {
	return model.space
}

// Status for simpleocmodel
func (model SimpleOCModel) Status() OCStatus {
	return model.status
}

// RuntimeFigures for simpleocmodel
func (model SimpleOCModel) RuntimeFigures() RuntimeFigures {
	return model.runtimeFigures
}

// Conf for simpleocmodel
func (model SimpleOCModel) Conf() Conf {
	return model.conf
}

// Impl for simpleocmodel
func (model SimpleOCModel) Impl() Impl {
	return model.impl
}

// NewSimpleOCModel creates a simple oc model
func NewSimpleOCModel(conf Conf, space Space, status OCStatus, runtimeFigures RuntimeFigures, centroids Clust) OCModel {
	return SimpleOCModel{
		conf:           conf,
		space:          space,
		status:         status,
		runtimeFigures: runtimeFigures,
		centroids:      centroids,
		impl:           nil,
	}
}
