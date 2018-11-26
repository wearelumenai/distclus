package core

// Impl concrete algorithms
type Impl interface {
	Init(conf Conf, space Space) error
	Run(conf Conf, space Space, closing <-chan bool) error
	Push(elemt Elemt) error
	SetAsync() error
	Centroids() (Clust, error)
}
