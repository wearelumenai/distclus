package core_test

import (
	"testing"

	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/euclid"
	"github.com/wearelumenai/distclus/v0/internal/test"
)

func TestClust_ParLosses(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	for degree := 1; degree < 100; degree++ {
		var seqLosses, seqCards = centroids.ReduceLoss(data, euclid.Space{}, 2.)
		var parLosses, parCards = centroids.ParReduceLoss(data, euclid.Space{}, 2., degree)

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
		var seqLoss = centroids.TotalLoss(data, euclid.Space{}, 2.)
		var parLoss = centroids.ParTotalLoss(data, euclid.Space{}, 2., degree)

		test.AssertAlmostEqual(t, seqLoss, parLoss)
	}
}

func TestClust_ParLossForLabels(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var labels = make([]int, 0, len(data))
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
		for j := 0; j < len(data); j++ {
			labels = append(labels, j%2)
		}
	}

	for degree := 1; degree < 100; degree++ {
		var seqLosses, seqCards = centroids.ReduceLossForLabels(data, labels, euclid.Space{}, 2.)
		var parLosses, parCards = centroids.ParReduceLossForLabels(data, labels, euclid.Space{}, 2., degree)

		test.AssertArrayAlmostEqual(t, seqLosses, parLosses)
		test.AssertArrayEqual(t, seqCards, parCards)
	}
}
