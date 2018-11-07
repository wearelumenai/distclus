package core

import (
	"errors"
	"fmt"
	"time"
)

type AlgorithmConf struct {
	Space Space
}

// Clustering algorithm abstract implementation
type AlgorithmTemplate struct {
	config          AlgorithmConf
	buffer          Buffer
	Clust           Clust
	status          ClustStatus
	closing         chan bool
	closed          chan bool
	templateMethods AlgorithmTemplateMethods
}

// Template methods to be implemented by concrete algorithms
type AlgorithmTemplateMethods struct {
	Initialize func() (centroids Clust, ready bool)
	Run        func(closing <-chan bool)
}

// Create a new algorithm instance
func NewAlgorithmTemplate(config AlgorithmConf, buffer Buffer, methods AlgorithmTemplateMethods) *AlgorithmTemplate {
	var algo = AlgorithmTemplate{}
	algo.config = config
	algo.buffer = buffer
	algo.templateMethods = methods
	algo.status = Created
	algo.closing = make(chan bool, 1)
	algo.closed = make(chan bool, 1)
	return &algo
}

// Get the centroids currently found by the algorithm
func (algo *AlgorithmTemplate) Centroids() (clust Clust, err error) {
	switch algo.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		clust = algo.Clust
	}
	return clust, err
}

// Push a new observation in the algorithm
func (algo *AlgorithmTemplate) Push(elemt Elemt) (err error) {
	switch algo.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		algo.buffer.Push(elemt)
	}
	return err
}

// Run the algorithm, asynchronously if async is true
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

// Predict the cluster for a new observation
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

// Stop the algorithm
func (mcmc *AlgorithmTemplate) Close() {
	if mcmc.status == Running {
		mcmc.closing <- true
		<-mcmc.closed
	}
	mcmc.status = Closed
}

// Initialize the algorithm, if success run it synchronously otherwise return an error
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

// Initialize the algorithm, if success run it asynchronously otherwise retry
func (algo *AlgorithmTemplate) initAndRunAsync() error {
	var err = algo.initAndRunSync()
	if err != nil {
		time.Sleep(300 * time.Millisecond)
		err = algo.initAndRunAsync()
	}
	return err
}
