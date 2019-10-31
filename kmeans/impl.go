package kmeans

import (
	"distclus/core"
	"distclus/figures"
	"fmt"
	"sync"
	"time"
)

// Impl algorithm abstract implementation
type Impl struct {
	strategy    Strategy
	buffer      core.Buffer
	initializer core.Initializer
	iter        int
	forever     bool
	wakeUp      chan bool
	paused      bool
	mutex       sync.RWMutex
	notifier    core.StatusNotifier
}

// Strategy Abstract Impl strategy to be implemented by concrete algorithms
type Strategy interface {
	Iterate(space core.Space, centroids core.Clust, data []core.Elemt) core.Clust
}

// Init Algorithm
func (impl *Impl) Init(conf core.ImplConf, space core.Space) (core.Clust, error) {
	var kmeansConf = conf.(Conf)
	_ = impl.buffer.Apply()
	impl.iter = 0
	return impl.initializer(kmeansConf.K, impl.buffer.Data(), space, kmeansConf.RGen)
}

// Run the algorithm until signal received on closing channel or iteration number is reached
func (impl *Impl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier core.Notifier, closing <-chan bool, closed chan<- bool) (err error) {
	var kmeansConf = conf.(Conf)
	for loop := impl.forever || impl.iter < kmeansConf.Iter; loop; {
		select {
		case <-closing:
			closed <- true
			time.Sleep(300 * time.Millisecond)
			loop = false

		default:
			impl.iter++
			if impl.forever {
				impl.mutex.RLock()
				if impl.paused {
					impl.mutex.RUnlock()
					var _, ok = <-impl.wakeUp
					if !ok {
						break
					}
				}
				impl.mutex.RUnlock()
			}
			centroids = impl.strategy.Iterate(space, centroids, impl.buffer.Data())
			notifier(centroids, impl.runtimeFigures())
			err = impl.buffer.Apply()
			loop = impl.forever || impl.iter < kmeansConf.Iter
		}
	}
	return
}

// Push input element in the buffer
func (impl *Impl) Push(elemt core.Elemt) error {
	return impl.buffer.Push(elemt)
}

// SetAsync changes the status of impl buffer to async
func (impl *Impl) SetAsync(notifier core.StatusNotifier) error {
	impl.notifier = notifier
	impl.forever = true
	impl.wakeUp = make(chan bool)
	return impl.buffer.SetAsync()
}

// runtimeFigures returns specific kmeans properties
func (impl *Impl) runtimeFigures() map[string]float64 {
	return map[string]float64{figures.Iterations: float64(impl.iter)}
}

// Pause asynchronous execution
func (impl *Impl) Pause() (err error) {
	if impl.forever {
		impl.mutex.Lock()
		impl.paused = true
		impl.mutex.Unlock()
	} else {
		err = fmt.Errorf("Batch mode")
	}
	return
}

// WakeUp cancel paused status
func (impl *Impl) WakeUp() (err error) {
	if impl.forever {
		impl.mutex.Lock()
		impl.paused = false
		impl.wakeUp <- true
		impl.mutex.Unlock()
	} else {
		err = fmt.Errorf("Batch mode")
	}
	return
}
