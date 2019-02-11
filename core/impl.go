package core

type Notifier = func(Clust, map[string]float64)

// Impl concrete algorithms
type Impl interface {
	Init(ImplConf, Space) (centroids Clust, err error)
	Run(conf ImplConf, space Space, centroids Clust, notifier Notifier, closing <-chan bool, closed chan<- bool) error
	Push(Elemt) error
	SetAsync() error
}

// ImplConf is implementation configuration interface
type ImplConf interface{}
