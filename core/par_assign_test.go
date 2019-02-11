package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/vectors"
	"reflect"
	"runtime"
	"testing"
)

func TestClust_ParAssignDBA(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.TestVectors)*20)
	var centroids = core.Clust(test.TestVectors[0:3])
	for i := 0; i < 20; i++ {
		data = append(data, test.TestVectors...)
	}

	var seq_dbas, seq_cards = centroids.AssignDBA(data, vectors.Space{})
	var par_dbas, par_cards = centroids.ParAssignDBA(data, vectors.Space{}, runtime.NumCPU())

	test.AssertCentroids(t, seq_dbas, par_dbas)

	if !reflect.DeepEqual(seq_cards, par_cards) {
		t.Error("cardinality error")
	}
}
