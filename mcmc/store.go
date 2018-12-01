package mcmc

import (
	"distclus/core"
	"distclus/kmeans"

	"golang.org/x/exp/rand"
)

// CenterStore structure
type CenterStore struct {
	centers map[int]core.Clust
	rgen    *rand.Rand
}

// NewCenterStore returns a new center store
func NewCenterStore(rgen *rand.Rand) CenterStore {
	return CenterStore{
		centers: map[int]core.Clust{},
		rgen:    rgen,
	}
}

// GetCenters returns input centroids centers
func (store *CenterStore) GetCenters(buffer core.Buffer, space core.Space, k int, clust core.Clust) core.Clust {
	var centers, ok = store.centers[k]

	if !ok {
		centers = store.genCenters(buffer, space, k, clust)
	}

	return centers
}

// SetCenters set centers to input centroids
func (store *CenterStore) SetCenters(clust core.Clust) {
	store.centers[len(clust)] = clust
}

func (store *CenterStore) genCenters(buffer core.Buffer, space core.Space, k int, prev core.Clust) (clust core.Clust) {
	var prevK = len(prev)

	switch {
	case prevK < k:
		clust = store.addCenter(buffer, space, prevK, prev)

	case prevK > k:
		clust = store.delCenter(space, prevK, prev)

	case prevK == k:
		clust = prev
	}

	return
}

func (store *CenterStore) addCenter(buffer core.Buffer, space core.Space, prevK int, prev core.Clust) (clust core.Clust) {
	clust = make(core.Clust, prevK+1)
	for i := 0; i < prevK; i++ {
		clust[i] = space.Copy(prev[i])
	}
	clust[prevK] = kmeans.PPIter(prev, buffer.Data(), space, store.rgen)
	return
}

func (store *CenterStore) delCenter(space core.Space, prevK int, prev core.Clust) (clust core.Clust) {
	var del = store.rgen.Intn(prevK)
	clust = make(core.Clust, prevK-1)
	for i := 0; i < prevK-1; i++ {
		if i < del {
			clust[i] = space.Copy(prev[i])
		} else {
			clust[i] = space.Copy(prev[i+1])
		}
	}
	return
}
