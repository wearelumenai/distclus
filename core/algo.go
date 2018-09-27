package core

import (
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
	"time"
)

type AlgoConf struct {
	InitK              int
	Space              Space
	FrameSize          int
	RGen               *rand.Rand
}

type AbstractAlgo struct {
	config AlgoConf
	Buffer
	Clust        Clust
	status       ClustStatus
	initializer  Initializer
	closing      chan bool
	closed       chan bool
	RunAlgorithm func(<-chan bool)
}

func NewAlgo(config AlgoConf, data []Elemt, initializer Initializer) *AbstractAlgo {
	var algo = AbstractAlgo{}
	algo.config = config
	algo.Buffer = NewBuffer(data, config.FrameSize)
	algo.initializer = initializer
	algo.status = Created
	algo.closing = make(chan bool, 1)
	algo.closed = make(chan bool, 1)
	return &algo
}

func (algo *AbstractAlgo) Centroids() (clust Clust, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		clust = algo.Clust
	}
	return clust, err
}

func (algo *AbstractAlgo) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		algo.Buffer.Push(elemt)
	}
	return err
}

func (algo *AbstractAlgo) Run(async bool) {
	if async {
		algo.Buffer.SetAsync()
		go algo.initAndRunAsync()
	} else {
		algo.initAndRunSync()
	}
}

func (algo *AbstractAlgo) Predict(elemt Elemt, push bool) (pred Elemt, idx int, err error) {
	var clust Clust
	clust, err = algo.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, algo.config.Space)
		if push {
			err = algo.Push(elemt)
		}
	}

	return pred, idx, err
}

func (mcmc *AbstractAlgo) Close() {
	mcmc.closing <- true
	<-mcmc.closed
}

func (algo *AbstractAlgo) initAndRunSync() error {
	var ok bool
	algo.Clust, ok = algo.initializer(algo.config.InitK, algo.Data, algo.config.Space, algo.config.RGen)
	if ok {
		algo.status = Running
		algo.RunAlgorithm(algo.closing)
		algo.status = Closed
		algo.closed <- true
		return nil
	}
	return errors.New("Failed to initialize")
}

func (algo *AbstractAlgo) initAndRunAsync() error {
	var err = algo.initAndRunSync()
	if err != nil {
		time.Sleep(300 * time.Millisecond)
		algo.Buffer.Apply()
		err = algo.initAndRunAsync()
	}
	return err
}


