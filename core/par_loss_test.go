package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/vectors"
	"reflect"
	"runtime"
	"testing"
)

func TestClust_ParLosses(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	var seqLosses, seqCards = centroids.Losses(data, vectors.Space{}, 2.)
	var parLosses, parCards = centroids.ParLosses(data, vectors.Space{}, 2., runtime.NumCPU())

	test.AssertArrayAlmostEqual(t, seqLosses, parLosses)

	if !reflect.DeepEqual(seqCards, parCards) {
		t.Error("cardinality error")
	}
}

func TestClust_ParLoss(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	var seqLoss = centroids.Loss(data, vectors.Space{}, 2.)
	var parLoss = centroids.ParLoss(data, vectors.Space{}, 2., runtime.NumCPU())

	test.AssertAlmostEqual(t, seqLoss, parLoss)
}
