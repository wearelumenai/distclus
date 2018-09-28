package core

import (
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
	"time"
)

type AlgorithmConf struct {
	InitK     int
	Space     Space
	FrameSize int
	RGen      *rand.Rand
}

type AlgorithmTemplate struct {
	config          AlgorithmConf
	buffer          Buffer
	Clust           Clust
	status          ClustStatus
	closing         chan bool
	closed          chan bool
	templateMethods AlgorithmTemplateMethods
}

type AlgorithmTemplateMethods struct {
	Initialize func() (centroids Clust, ready bool)
	Run        func(closing <-chan bool)
}

func NewAlgo(config AlgorithmConf, buffer Buffer, templateMethods AlgorithmTemplateMethods) *AlgorithmTemplate {
	var algo = AlgorithmTemplate{}
	algo.config = config
	algo.buffer = buffer
	algo.templateMethods = templateMethods
	algo.status = Created
	algo.closing = make(chan bool, 1)
	algo.closed = make(chan bool, 1)
	return &algo
}

func (algo *AlgorithmTemplate) Centroids() (clust Clust, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		clust = algo.Clust
	}
	return clust, err
}

func (algo *AlgorithmTemplate) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		algo.buffer.Push(elemt)
	}
	return err
}

func (algo *AlgorithmTemplate) Run(async bool) {
	if async {
		algo.buffer.SetAsync()
		go algo.initAndRunAsync()
	} else {
		var err = algo.initAndRunSync()
		if err != nil {
			panic(err)
		}
	}
	algo.status = Running
}

func (algo *AlgorithmTemplate) Predict(elemt Elemt, push bool) (pred Elemt, label int, err error) {
	var clust Clust
	clust, err = algo.Centroids()

	if err == nil {
		pred, label, _ = clust.Assign(elemt, algo.config.Space)
		if push {
			err = algo.Push(elemt)
		}
	}

	return pred, label, err
}

func (mcmc *AlgorithmTemplate) Close() {
	mcmc.closing <- true
	<-mcmc.closed
	mcmc.status = Closed
}

func (algo *AlgorithmTemplate) initAndRunSync() error {
	var ok bool
	algo.Clust, ok = algo.templateMethods.Initialize()
	if ok {
		algo.templateMethods.Run(algo.closing)
		algo.closed <- true
		return nil
	}
	return errors.New("Failed to initialize")
}

func (algo *AlgorithmTemplate) initAndRunAsync() error {
	var err = algo.initAndRunSync()
	if err != nil {
		time.Sleep(300 * time.Millisecond)
		err = algo.initAndRunAsync()
	}
	return err
}
