package streaming

import (
	"distclus/core"
	"errors"
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
	async       bool
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
			Mu:    GetRadius(conf.Lambda),
			Sigma: .2,
			Src:   conf.RGen,
		},
	}
}

// Init initializes the streaming algorithm.
func (impl *Impl) Init(core.ImplConf, core.Space) (core.Clust, error) {
	select {
	case centroids := <-impl.c:
		return core.Clust{centroids}, nil
	default:
		return nil, errors.New("at least one element is needed")
	}
}

// Run runs the streaming algorithm.
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier core.Notifier, closing <-chan bool, closed chan<- bool) error {
	for i := range centroids {
		impl.AddCenter(centroids[i], 0.)
	}
	for loop := true; loop; {
		if impl.async {
			loop = impl.iterAsync(space, notifier, closing)
		} else {
			loop = impl.iterSync(space, notifier)
		}
	}
	closed <- true
	return nil
}

func (impl *Impl) iterAsync(space core.Space, notifier core.Notifier, closing <-chan bool) bool {
	select {
	case elemt := <-impl.c:
		impl.iter(elemt, space, notifier)
		return true
	case <-closing:
		return false
	}
}

func (impl *Impl) iterSync(space core.Space, notifier core.Notifier) bool {
	select {
	case elemt := <-impl.c:
		impl.iter(elemt, space, notifier)
		return true
	default:
		return false
	}
}

func (impl *Impl) iter(elemt core.Elemt, space core.Space, notifier core.Notifier) {
	impl.Iterate(elemt, space)
	notifier(impl.clust, nil)
}

// Push pushes a new element
func (impl *Impl) Push(elemt core.Elemt) error {
	if impl.async {
		return impl.pushAsync(elemt)
	}
	return impl.pushSync(elemt)
}

func (impl *Impl) pushAsync(elemt core.Elemt) error {
	impl.c <- elemt
	return nil
}

func (impl *Impl) pushSync(elemt core.Elemt) error {
	select {
	case impl.c <- elemt:
		return nil
	default:
		return errors.New("buffer is full")
	}
}

// SetAsync indicates that the algorithm is asynchronous
func (impl *Impl) SetAsync() error {
	impl.async = true
	return nil
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
	if distance < impl.maxDistance {
		return distance / impl.maxDistance
	}
	return 1
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

// Iterate process a streaming iteration.
func (impl *Impl) Iterate(elemt core.Elemt, space core.Space) {
	var _, label, distance = impl.clust.Assign(elemt, space)
	var relative = impl.GetRelativeDistance(distance)

	if impl.count < 5 || relative < impl.conf.B {
		var threshold = impl.norm.Rand()
		if threshold < relative {
			impl.AddCenter(elemt, distance)
		} else {
			impl.UpdateCenter(label, elemt, distance, space)
		}
	} else {
		impl.AddOutlier(elemt)
	}
	impl.count++
}

// GetRadius returns the radius of the sphere that might contains new centers
func GetRadius(Lambda float64) float64 {
	return 1.1 - Lambda*.1
}
