package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/vectors"
	"testing"
)

func TestClust_ParLosses(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	for degree := 1; degree < 100; degree++ {
		var seqLosses, seqCards = centroids.ReduceLoss(data, vectors.Space{}, 2.)
		var parLosses, parCards = centroids.ParReduceLoss(data, vectors.Space{}, 2., degree)

		test.AssertArrayAlmostEqual(t, seqLosses, parLosses)
		test.AssertArrayEqual(t, seqCards, parCards)
	}
}

func TestClust_ParLoss(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	for degree := 1; degree < 100; degree++ {
		var seqLoss = centroids.TotalLoss(data, vectors.Space{}, 2.)
		var parLoss = centroids.ParTotalLoss(data, vectors.Space{}, 2., degree)

		test.AssertAlmostEqual(t, seqLoss, parLoss)
	}
}
