package core

// Impl concrete algorithms
type Impl interface {
	Init(ImplConf, Space) (centroids Clust, err error)
	Run(conf ImplConf, space Space, centroids Clust, notifier func(Clust), closing <-chan bool, closed chan<- bool) error
	Push(Elemt) error
	SetAsync() error
	Iterations() int
}

// ImplConf is implementation configuration interface
type ImplConf interface{}
