package streaming_test

import (
	"distclus/core"
	"distclus/streaming"
	"distclus/vectors"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
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
}

func TestImpl_GetRadius(t *testing.T) {
	if radius := streaming.GetRadius(1.); radius != 1. {
		t.Error("expected 1. got", radius)
	}
	if radius := streaming.GetRadius(.1); radius != 1.09 {
		t.Error("expected 1.09 got", radius)
	}
}

func TestImpl_Interface(t *testing.T) {
	var impl interface{} = &streaming.Impl{}
	var _, ok = impl.(core.Impl)
	if !ok {
		t.Error("core.Impl should be implemented")
	}
}

func TestImpl_InitError(t *testing.T) {
	var impl = streaming.Impl{}
	var _, err = impl.Init(streaming.Conf{}, vectors.Space{})
	if err == nil {
		t.Error("an error was expected (initialization is not possible)")
	}
}

func TestImpl_InitSuccess(t *testing.T) {
	var impl = streaming.NewImpl(streaming.Conf{})
	var cluster0 = []float64{1.}
	var err0 = impl.Push(cluster0)
	if err0 != nil {
		t.Error("unexpected error", err0)
	}
	var clust, err = impl.Init(streaming.Conf{}, vectors.Space{})
	if err != nil {
		t.Error("unexpected error", err)
	}
	if !reflect.DeepEqual(core.Clust{cluster0}, clust) {
		t.Error("initialization failed")
	}
}

func TestImpl_PushError(t *testing.T) {
	var conf = streaming.Conf{}
	var impl = streaming.NewImpl(conf)

	var cluster0 = []float64{1.}

	streaming.SetConfigDefaults(&conf)
	for i := 0; i < conf.BufferSize; i++ {
		var _ = impl.Push(cluster0)
	}

	var err0 = impl.Push(cluster0)
	if err0 == nil {
		t.Error("an error was expected (channel is full)")
	}
}

func TestImpl_Iterate(t *testing.T) {
	var r = mix()
	var impl = streaming.NewImpl(streaming.Conf{})
	impl.AddCluster(r(), 0.)
	for i := 0; i < 1000; i++ {
		var cluster1 = r()
		impl.Iterate(cluster1, vectors.Space{})
	}
	var clusters = impl.GetClusters()
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c)
	}
	if len(clusters) > 6 {
		t.Error("less than 6 clusters expected")
	}
}

func mix() func() []float64 {
	var norm1, _ = distmv.NewNormal([]float64{1., 1.}, mat.NewDiagDense(2, []float64{2., 2.}), nil)
	var norm2, _ = distmv.NewNormal([]float64{-23., 9.}, mat.NewDiagDense(2, []float64{4., 2.}), nil)
	var norm3, _ = distmv.NewNormal([]float64{-12., -25.}, mat.NewDiagDense(2, []float64{2., 4.}), nil)
	var p = distuv.Uniform{
		Min: 0.,
		Max: 1.,
	}
	return func() []float64 {
		switch a := p.Rand(); {
		case a < .2:
			return norm1.Rand(nil)
		case a < .5:
			return norm2.Rand(nil)
		default:
			return norm3.Rand(nil)
		}
	}
}
