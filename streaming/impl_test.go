package streaming_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/streaming"
	"golang.org/x/exp/rand"
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
	var impl = streaming.NewImpl(streaming.Conf{}, nil)
	var relDist = impl.GetRelativeDistance(1.2)
	if relDist != 1. {
		t.Error("expected 1. got", relDist)
	}

	impl.UpdateMaxDistance(1.2)
	if relDist := impl.GetRelativeDistance(0.6); relDist != 0.5 {
		t.Error("expected 0.5 got", relDist)
	}

	if relDist := impl.GetRelativeDistance(1.5); relDist != 1.25 {
		t.Error("expected 1.25 got", relDist)
	}
}

func TestImpl_AddCenter(t *testing.T) {
	var impl = streaming.Impl{}

	var cluster0 = []float64{1.}
	impl.AddCenter(cluster0, 1.2)

	if c0 := impl.GetClusters()[0]; !reflect.DeepEqual(cluster0, c0) {
		t.Error("expected cluster: ", cluster0)
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}

	var cluster1 = []float64{2.}
	impl.AddCenter(cluster1, 1.1)
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
	impl.AddCenter(cluster0, 1.2)
	var cluster1 = []float64{2.}
	impl.AddOutlier(cluster1)
	if c1 := impl.GetClusters()[1]; !reflect.DeepEqual(cluster1, c1) {
		t.Error("expected cluster: ", cluster1)
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.2 {
		t.Error("expected 1.2 got", maxDist)
	}
}

func TestImpl_UpdateCenter(t *testing.T) {
	var impl = streaming.Impl{}

	impl.AddCenter(core.Elemt([]float64{1.}), 1.2)
	impl.UpdateCenter(0, core.Elemt([]float64{2.}), 1.3, euclid.Space{})
	impl.UpdateCenter(0, core.Elemt([]float64{3.}), 1.1, euclid.Space{})
	if c0 := impl.GetClusters()[0]; !reflect.DeepEqual([]float64{2.}, c0) {
		t.Error("expected cluster: ", []float64{2.})
	}
	if maxDist := impl.GetMaxDistance(); maxDist != 1.3 {
		t.Error("expected 1.3 got", maxDist)
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
	var _, err = impl.Init(streaming.Conf{}, euclid.Space{})
	if err == nil {
		t.Error("an error was expected (initialization is not possible)")
	}
}

func TestImpl_InitSuccess(t *testing.T) {
	var conf = streaming.Conf{BufferSize: 5}
	var impl = streaming.NewImpl(conf, []core.Elemt{})
	var cluster0 = []float64{1.}
	var err0 = impl.Push(cluster0)
	if err0 != nil {
		t.Error("unexpected error", err0)
	}
	var clust, err = impl.Init(conf, euclid.Space{})
	if err != nil {
		t.Error("unexpected error", err)
	}
	if !reflect.DeepEqual(core.Clust{cluster0}, clust) {
		t.Error("initialization failed")
	}
}

func TestImpl_PushError(t *testing.T) {
	var conf = streaming.Conf{}
	var impl = streaming.NewImpl(conf, []core.Elemt{})

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

var conf = streaming.Conf{
	BufferSize: 5, Mu: .5, Sigma: .1, OutRatio: 2., OutAfter: 5,
	RGen: rand.New(rand.NewSource(1514613616431)),
}

func TestImpl_Iterate(t *testing.T) {
	var distr = mix()
	var impl = streaming.NewImpl(conf, []core.Elemt{})
	impl.AddCenter(distr(), 0.)
	for i := 0; i < 1000; i++ {
		var cluster1 = distr()
		impl.Iterate(cluster1, euclid.Space{})
	}
	var clusters = impl.GetClusters()
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c)
	}
	if len(clusters) > 6 {
		t.Error("less than 6 clusters expected")
	}
}

func TestImpl_Run(t *testing.T) {
	var distr = mix()
	var impl = streaming.NewImpl(conf, []core.Elemt{})
	var closing = make(chan bool, 1)
	var closed = make(chan bool, 1)
	var clusters core.Clust
	var notifier = func(clusts core.Clust, float64s map[string]float64) {
		clusters = clusts
	}
	_ = impl.SetAsync()
	go func() {
		_ = impl.Run(conf, euclid.Space{}, core.Clust{distr()}, notifier, closing, closed)
	}()
	for i := 0; i < 1000; i++ {
		_ = impl.Push(distr())
	}
	closing <- true
	<-closed
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c)
	}
	if len(clusters) > 6 {
		t.Error("less than 6 clusters expected")
	}
}
