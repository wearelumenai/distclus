package streaming

import (
	"distclus/core"
	"distclus/kmeans"
	"golang.org/x/exp/rand"
)

type CenterStore struct {
	centers map[int]core.Clust
	buffer *core.DataBuffer
	space core.Space
	rgen *rand.Rand
}

func NewCenterStore(buffer *core.DataBuffer, space core.Space, rgen *rand.Rand) CenterStore {
	var store = CenterStore{}
	store.centers = make(map[int]core.Clust)
	store.buffer = buffer
	store.space = space
	store.rgen = rgen
	return store
}

func (store *CenterStore) GetCenters(k int, clust core.Clust) core.Clust {
	var centers, ok = store.centers[k]

	if !ok {
		centers = store.genCenters(k, clust)
	}

	return centers
}

func (store *CenterStore) SetCenters(clust core.Clust) {
	store.centers[len(clust)] = clust
}

func (store *CenterStore) genCenters(k int, prev core.Clust) core.Clust {
	var prevK = len(prev)
	var clust core.Clust

	switch {
	case prevK < k:
		clust = store.addCenter(prevK, prev)

	case prevK > k:
		clust = store.delCenter(prevK, prev)

	case prevK == k:
		clust = prev
	}

	return clust
}

func (store *CenterStore) addCenter(prevK int, prev core.Clust) core.Clust {
	var clust = make(core.Clust, prevK+1)
	for i := 0; i < prevK; i++ {
		clust[i] = store.space.Copy(prev[i])
	}
	clust[prevK] = kmeans.KMeansPPIter(prev, store.buffer.Data, store.space, store.rgen)
	return clust
}

func (store *CenterStore) delCenter(prevK int, prev core.Clust) core.Clust {
	var del = store.rgen.Intn(prevK)
	var clust = make(core.Clust, prevK-1)
	for i := 0; i < prevK-1; i++ {
		if i < del {
			clust[i] = store.space.Copy(prev[i])
		} else {
			clust[i] = store.space.Copy(prev[i+1])
		}
	}
	return clust
}
