package streaming

import (
	"distclus/core"
)

type Impl struct {
	maxDistance float64
	clust       core.Clust
	cards       []int
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
