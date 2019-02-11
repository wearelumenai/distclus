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
	var data = make([]core.Elemt, 0, len(test.TestVectors)*20)
	var centroids = core.Clust(test.TestVectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.TestVectors...)
	}

	var seq_losses, seq_cards = centroids.Losses(data, vectors.Space{}, 2.)
	var par_losses, par_cards = centroids.ParLosses(data, vectors.Space{}, 2., runtime.NumCPU())

	test.AssertArrayAlmostEqual(t, seq_losses, par_losses)

	if !reflect.DeepEqual(seq_cards, par_cards) {
		t.Error("cardinality error")
	}
}

func TestClust_ParLoss(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.TestVectors)*20)
	var centroids = core.Clust(test.TestVectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.TestVectors...)
	}

	var seq_loss = centroids.Loss(data, vectors.Space{}, 2.)
	var par_loss = centroids.ParLoss(data, vectors.Space{}, 2., runtime.NumCPU())

	test.AssertAlmostEqual(t, seq_loss, par_loss)
}
