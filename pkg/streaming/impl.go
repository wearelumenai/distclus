package streaming

import (
	"errors"
	"github.com/wearelumenai/distclus/v0/pkg/core"
	"github.com/wearelumenai/distclus/v0/pkg/figures"

	"gonum.org/v1/gonum/stat/distuv"
)

// Impl represents the implementation of the streaming clustering algorithm.
type Impl struct {
	maxDistance float64
	clust       core.Clust
	cards       []int
	c           chan core.Elemt
	conf        Conf
	norm        distuv.Normal
	count       int
}

// Copy impl
func (impl *Impl) Copy(conf core.ImplConf, space core.Space) (core.Impl, error) {
	var newConf = conf.(*Conf)
	var algo = NewAlgo(*newConf, space, impl.clust)
	return algo.Impl(), nil
}

// NewImpl creates a new Impl instance.
func NewImpl(conf Conf, elemts []core.Elemt) Impl {
	var c = make(chan core.Elemt, conf.BufferSize)
	for i := range elemts {
		c <- elemts[i]
	}
	return Impl{
		c:    c,
		conf: conf,
		norm: distuv.Normal{
			Mu:    conf.Mu,
			Sigma: conf.Sigma,
			Src:   conf.RGen,
		},
	}
}

// Init initializes the streaming algorithm.
func (impl *Impl) Init(_ core.ImplConf, _ core.Space, _ core.Clust) (clust core.Clust, err error) {
	select {
	case centroids := <-impl.c:
		clust = core.Clust{centroids}
		for i := range clust {
			impl.AddCenter(clust[i], 0.)
		}
	default:
		err = errors.New("at least one element is needed")
	}
	return
}

// Iterate runs the streaming algorithm.
func (impl *Impl) Iterate(conf core.ImplConf, space core.Space, centroids core.Clust) (clust core.Clust, runtimeFigures figures.RuntimeFigures, err error) {
	select {
	case elemt := <-impl.c:
		impl.Process(elemt, space)
		clust = impl.clust
	default:
	}
	runtimeFigures = impl.runtimeFigures()
	return
}

func (impl *Impl) runtimeFigures() figures.RuntimeFigures {
	return figures.RuntimeFigures{figures.MaxDistance: impl.maxDistance}
}

// Push pushes a new element
func (impl *Impl) Push(elemt core.Elemt, running bool) (err error) {
	select {
	case impl.c <- elemt:
	default:
		err = errors.New("buffer is full")
	}
	return
}

// UpdateMaxDistance changes the maximal distance between two clusters
func (impl *Impl) UpdateMaxDistance(distance float64) {
	if distance > impl.maxDistance {
		impl.maxDistance = distance
	}
}

// GetMaxDistance returns the maximal distance between two clusters
func (impl *Impl) GetMaxDistance() float64 {
	return impl.maxDistance
}

// GetRelativeDistance returns the ratio of given distance with the maximal distance if less than 1, otherwise 1.
func (impl *Impl) GetRelativeDistance(distance float64) float64 {
	if impl.maxDistance == 0 {
		impl.maxDistance = distance
	}
	return distance / impl.maxDistance
}

// AddCenter adds a new center.
func (impl *Impl) AddCenter(cluster core.Elemt, distance float64) {
	impl.clust = append(impl.clust, cluster)
	impl.cards = append(impl.cards, 1)
	impl.UpdateMaxDistance(distance)
}

// AddOutlier adds an outlier.
func (impl *Impl) AddOutlier(outlier core.Elemt) {
	impl.clust = append(impl.clust, outlier)
	impl.cards = append(impl.cards, 1)
}

// UpdateCenter modifies an existing center.
func (impl *Impl) UpdateCenter(label int, elemt core.Elemt, distance float64, space core.Space) {
	var cluster = space.Combine(impl.clust[label], impl.cards[label], elemt, 1)
	impl.clust[label] = cluster
	impl.cards[label]++
	impl.UpdateMaxDistance(distance)
}

// GetClusters returns the current cluster centers.
func (impl *Impl) GetClusters() core.Clust {
	return impl.clust
}

// Process a streaming iteration.
func (impl *Impl) Process(elemt core.Elemt, space core.Space) {
	var _, label, distance = impl.clust.Assign(elemt, space)
	var relative = impl.GetRelativeDistance(distance)

	if impl.count >= impl.conf.OutAfter && relative > impl.conf.OutRatio {
		impl.AddOutlier(elemt)
	} else {
		var threshold = impl.norm.Rand()
		if threshold < relative {
			impl.AddCenter(elemt, distance)
		} else {
			impl.UpdateCenter(label, elemt, distance, space)
		}
	}
	impl.count++
}
