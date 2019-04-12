package streaming

import (
	"distclus/core"
	"distclus/vectors"
	"errors"
	"gonum.org/v1/gonum/stat/distuv"
)

type Impl struct {
	maxDistance float64
	clust       core.Clust
	cards       []int
	c           chan core.Elemt
	conf        Conf
	norm        distuv.Normal
	count       int
}

func NewImpl(conf Conf) *Impl {
	SetConfigDefaults(&conf)
	return &Impl{
		c:    make(chan core.Elemt, conf.BufferSize),
		conf: conf,
		norm: distuv.Normal{
			Mu:    GetRadius(conf.Lambda),
			Sigma: .2,
			Src:   conf.RGen,
		},
	}
}

func (impl *Impl) Init(core.ImplConf, core.Space) (core.Clust, error) {
	select {
	case centroids := <-impl.c:
		return core.Clust{centroids}, nil
	default:
		return nil, errors.New("at least one element is needed")
	}
}

func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier core.Notifier, closing <-chan bool, closed chan<- bool) error {
	panic("implement me")
}

func (impl *Impl) Push(elemt core.Elemt) error {
	select {
	case impl.c <- elemt:
		return nil
	default:
		return errors.New("buffer is full")
	}
}

func (impl *Impl) SetAsync() error {
	return nil
}

func (impl *Impl) UpdateMaxDistance(distance float64) {
	if distance > impl.maxDistance {
		impl.maxDistance = distance
	}
}

func (impl *Impl) GetMaxDistance() float64 {
	return impl.maxDistance
}

func (impl *Impl) GetRelativeDistance(distance float64) float64 {
	if distance < impl.maxDistance {
		return distance / impl.maxDistance
	}
	return 1
}

func (impl *Impl) AddCluster(cluster core.Elemt, distance float64) {
	impl.clust = append(impl.clust, cluster)
	impl.cards = append(impl.cards, 1)
	impl.UpdateMaxDistance(distance)
}

func (impl *Impl) AddOutlier(outlier core.Elemt) {
	impl.clust = append(impl.clust, outlier)
	impl.cards = append(impl.cards, 1)
}

func (impl *Impl) UpdateCluster(label int, elemt core.Elemt, distance float64, space core.Space) {
	var cluster = space.Combine(impl.clust[label], impl.cards[label], elemt, 1)
	impl.clust[label] = cluster
	impl.cards[label] += 1
	impl.UpdateMaxDistance(distance)
}

func (impl *Impl) GetClusters() core.Clust {
	return impl.clust
}

func (impl *Impl) Iterate(elemt core.Elemt, space vectors.Space) {
	var _, label, distance = impl.clust.Assign(elemt, space)
	var relative = impl.GetRelativeDistance(distance)

	if impl.count < 5 || relative < impl.conf.B {
		var threshold = impl.norm.Rand()
		if threshold < relative {
			impl.AddCluster(elemt, distance)
		} else {
			impl.UpdateCluster(label, elemt, distance, space)
		}
	} else {
		impl.AddOutlier(elemt)
	}
}

func GetRadius(Lambda float64) float64 {
	return 1.1 - Lambda*.1
}
