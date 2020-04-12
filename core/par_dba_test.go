package core_test

import (
	"testing"

	"go.lumenai.fr/distclus/v0/core"
	"go.lumenai.fr/distclus/v0/euclid"
	"go.lumenai.fr/distclus/v0/internal/test"
)

func TestClust_ParReduceDBA(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	var centroids = core.Clust(test.Vectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}

	for degree := 1; degree < 100; degree++ {
		var seqDbas, seqCards = centroids.ReduceDBA(data, euclid.Space{})
		var parDbas, parCards = centroids.ParReduceDBA(data, euclid.Space{}, degree)

		test.AssertCentroids(t, seqDbas, parDbas)
		test.AssertArrayEqual(t, seqCards, parCards)
	}
}
