package streaming_test

import (
	"distclus/core"
	"distclus/streaming"
	"distclus/vectors"
	"reflect"
	"testing"
)

func TestImpl_UpdateMaxDistance(t *testing.T) {
	var impl = streaming.Impl{}
	impl.UpdateMaxDistance(1.2)
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}

	impl.UpdateMaxDistance(1.3)
	if maxDist := impl.GetMaxDistance(); maxDist != 1.3 {
		t.Error("expected 1.3 got", maxDist)
	}

	impl.UpdateMaxDistance(1.2)
	if maxDist := impl.GetMaxDistance(); maxDist != 1.3 {
		t.Error("expected 1.3 got", maxDist)
	}
}

func TestImpl_GetRelativeDistance(t *testing.T) {
	var impl = streaming.Impl{}
	var relDist = impl.GetRelativeDistance(1.2)
	if relDist != 1 {
		t.Error("expected 1 got", relDist)
	}

	impl.UpdateMaxDistance(1.2)
	if relDist := impl.GetRelativeDistance(0.6); relDist != 0.5 {
		t.Error("expected 0.5 got", relDist)
	}

	if relDist := impl.GetRelativeDistance(1.5); relDist != 1 {
		t.Error("expected 1 got", relDist)
	}
}

func TestImpl_AddCluster(t *testing.T) {
	var impl = streaming.Impl{}

	var cluster0 = []float64{1.}
	impl.AddCluster(cluster0, 1.2)

	if c0 := impl.GetClusters()[0]; !reflect.DeepEqual(cluster0, c0) {
		t.Error("expected cluster: ", cluster0)
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}

	var cluster1 = []float64{2.}
	impl.AddCluster(cluster1, 1.1)
	if c1 := impl.GetClusters()[1]; !reflect.DeepEqual(cluster1, c1) {
		t.Error("expected cluster: ", cluster1)
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}
}

func TestImpl_AddOutlier(t *testing.T) {
	var impl = streaming.Impl{}

	var cluster0 = []float64{1.}
	impl.AddCluster(cluster0, 1.2)
	var cluster1 = []float64{2.}
	impl.AddOutlier(cluster1)
	if c1 := impl.GetClusters()[1]; !reflect.DeepEqual(cluster1, c1) {
		t.Error("expected cluster: ", cluster1)
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}
}

func TestImpl_UpdateCluster(t *testing.T) {
	var impl = streaming.Impl{}

	impl.AddCluster(core.Elemt([]float64{1.}), 1.2)
	impl.UpdateCluster(0, core.Elemt([]float64{2.}), 1.3, vectors.Space{})
	impl.UpdateCluster(0, core.Elemt([]float64{3.}), 1.1, vectors.Space{})
	if c0 := impl.GetClusters()[0]; !reflect.DeepEqual([]float64{2.}, c0) {
		t.Error("expected cluster: ", []float64{2.})
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.3 {
		t.Error("expected 1.3 got", maxDist)
	}

	impl.AddOutlier(core.Elemt([]float64{1.}))
	impl.UpdateCluster(1, core.Elemt([]float64{2.}), 1.3, vectors.Space{})
	impl.UpdateCluster(1, core.Elemt([]float64{3.}), 1.1, vectors.Space{})
	if c0 := impl.GetClusters()[0]; !reflect.DeepEqual([]float64{2.}, c0) {
		t.Error("expected cluster: ", []float64{2.})
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.3 {
		t.Error("expected 1.3 got", maxDist)
	}
}
