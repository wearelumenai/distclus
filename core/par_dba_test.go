package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/vectors"
	"runtime"
	"testing"
)

func TestClust_ParReduceDBA(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	var seqDbas, seqCards = centroids.ReduceDBA(data, vectors.Space{})
	var parDbas, parCards = centroids.ParReduceDBA(data, vectors.Space{}, runtime.NumCPU())

	test.AssertCentroids(t, seqDbas, parDbas)
	test.AssertArrayEqual(t, seqCards, parCards)
}
