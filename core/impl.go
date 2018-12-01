package core

// Impl concrete algorithms
type Impl interface {
	Init(conf Conf, space Space) (centroids Clust, err error)
	Run(conf Conf, space Space, centroids Clust, notifier func(Clust), closing <-chan bool) error
	Push(elemt Elemt) error
	SetAsync() error
}
