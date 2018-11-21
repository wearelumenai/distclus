package streaming_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/streaming"
	"distclus/internal/test"
	"golang.org/x/exp/rand"
	"math"
	"runtime"
	"testing"
)

func TestStreaming_ParPredict_Given(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 0
	var algo = streaming.NewParStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, algo)
}

func TestStreaming_ParPredictKMeansPP(t *testing.T) {
	var conf = streamingConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232592652256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var algo = streaming.NewParStreaming(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestRunSyncKMeansPP(t, algo)
}

func TestStreaming_ParRunAsync(t *testing.T) {
	var conf = streamingConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.StreamingIter = 1 << 30
	var algo = streaming.NewParStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, algo)
}

func TestParStreamingStrategy_Loss(t *testing.T) {
	var conf = streamingConf
	conf.StreamingIter = 0
	var algo = streaming.NewParStreaming(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	test.PushAndRunSync(algo)


	var strategy = streaming.ParStreamingStrategy{}
	buffer := core.NewDataBuffer(test.TestVectors, conf.FrameSize)
	strategy.Buffer = buffer
	strategy.Config = conf
	strategy.Degree = runtime.NumCPU()

	var clust, _ = algo.Centroids()
	var l1 = strategy.Loss(clust)
	var l2 = clust.Loss(test.TestVectors, conf.Space, conf.Norm)

	if math.Abs(l1-l2)>1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}
