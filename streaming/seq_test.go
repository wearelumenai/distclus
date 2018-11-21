package streaming_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/streaming"
	"distclus/real"
	"golang.org/x/exp/rand"
	"testing"
)

var streamingConf = streaming.StreamingConf{
	AlgorithmConf: core.AlgorithmConf{
		Space: real.RealSpace{},
	},
	InitK: 3,
	FrameSize: 8,
	RGen:  rand.New(rand.NewSource(6305689164243)),
	Dim: 5, B: 100, Amp: 1,
	Norm: 2, Nu: 3, StreamingIter: 20,
	InitIter: 0,
}

var distrib = streaming.NewMultivT(streaming.MultivTConf{streamingConf})

func TestStreaming_Initialization(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 0
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, algo)
}

func TestStreaming_DefaultConf(t *testing.T) {
	var conf = streamingConf
	conf.RGen = nil
	conf.StreamingIter = 0
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, algo)
}

func TestStreaming_RunSyncGiven(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 0
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, algo)
}

func TestStreaming_RunSyncKMeansPP(t *testing.T) {
	var conf = streamingConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232592652256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestRunSyncKMeansPP(t, algo)
}

func TestStreaming_RunAsync(t *testing.T) {
	var conf = streamingConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.StreamingIter = 1 << 30
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, algo)
}

func TestStreaming_Workflow(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 1 << 30
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestWorkflow(t, algo)
}

func TestStreaming_MaxK(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var algo = streaming.NewSeqStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.PushAndRunSync(algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func TestStreaming_AcceptRatio(t *testing.T) {
	var algo = streaming.NewSeqStreaming(streamingConf, distrib, kmeans.GivenInitializer, []core.Elemt{})
	test.PushAndRunSync(algo)
	var r = algo.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}
