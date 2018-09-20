package kmeans

import (
	"fmt"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
	"errors"
)

type KMeans struct {
	KMeansSupport
	core.Buffer
	config KMeansConf
	status      core.ClustStatus
	initializer core.Initializer
	clust       core.Clust
	closing     chan bool
	closed      chan bool
}

type KMeansSupport interface {
	Iterate(proposal core.Clust) core.Clust
}

func NewSeqKMeans(conf KMeansConf, initializer core.Initializer, data []core.Elemt) *KMeans {
	conf.Verify()
	setConfigDefaults(&conf)

	var km KMeans
	km.config = conf
	km.initializer = initializer
	km.status = core.Created
	km.closing = make(chan bool, 1)
	km.closed = make(chan bool, 1)
	km.Buffer = core.NewBuffer(data, -1)
	km.KMeansSupport = SeqKMeansSupport{ buffer: &km.Buffer, config:km.config}

	return &km
}

func setConfigDefaults(conf *KMeansConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
}

func (km *KMeans) Centroids() (c core.Clust, err error) {
	switch km.status {
	case core.Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = km.clust
	}

	return
}

func (km *KMeans) Push(elemt core.Elemt) (err error) {
	switch km.status {
	case core.Closed:
		err = errors.New("clustering ended")
	default:
		km.Buffer.Push(elemt)
	}

	return err
}

func (km *KMeans) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int

	var clust, err = km.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, km.config.Space)
		if push {
			err = km.Push(elemt)
		}
	}

	return pred, idx, err
}

func (km *KMeans) Run(async bool) {
	if async {
		km.Buffer.SetAsync()
		go km.initAndRun(async)
	} else {
		km.initAndRun(async)
	}
}

func (km *KMeans) initAndRun(async bool) {
	for ok := false; !ok; {
		km.clust, ok = km.initializer(km.config.K, km.Data, km.config.Space, km.config.RGen)
		if !ok {
			km.handleFailedInitialisation(async)
		}
	}
	km.status = core.Running
	km.runAlgorithm()
}

func (km *KMeans) handleFailedInitialisation(async bool) {
	if !async {
		panic("failed to initialize")
	}
	time.Sleep(300 * time.Millisecond)
	km.Buffer.Apply()
}

func (km *KMeans) runAlgorithm() {
	for iter, loop := 0, true; iter < km.config.Iter && loop; iter++ {
		select {

		case <-km.closing:
			loop = false

		default:
			km.clust = km.Iterate(km.clust)
			km.Buffer.Apply()
		}
	}

	km.status = core.Closed
	km.closed <- true
}

func (km *KMeans) Close() {
	km.closing <- true
	<-km.closed
}

type SeqKMeansSupport struct {
	config KMeansConf
	buffer *core.Buffer
}

func (support SeqKMeansSupport) Iterate(clust core.Clust) core.Clust {

	var result, _ = clust.AssignDBA(support.buffer.Data, support.config.Space)

	return support.buildResult(clust, result)
}

func (support SeqKMeansSupport) buildResult(clust core.Clust, result core.Clust) core.Clust {
	for i := 0; i < len(result); i++ {
		if result[i] == nil {
			result[i] = clust[i]
		}
	}

	return result
}
