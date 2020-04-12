package core

import (
	"sync"
	"time"

	"github.com/wearelumenai/distclus/figures"
)

// Timeout interface for managing timeout algorithm
type Timeout interface {
	Disable()
	Enabled() bool
}

// timeout algorithm timeout structure
type timeout struct {
	duration         time.Duration
	enabled          bool
	ack              <-chan bool
	interruption     func(ClustStatus, error) error
	mutex            sync.RWMutex
	maxIter          int
	iterationsGetter func() int
}

func iterationsGetter(runtimeFiguresGetter func() (figures.RuntimeFigures, error)) func() int {
	return func() int {
		var rf, _ = runtimeFiguresGetter()
		return int(rf[figures.Iterations])
	}
}

func (timeout *timeout) Enabled() bool {
	timeout.mutex.RLock()
	defer timeout.mutex.RUnlock()
	return timeout.enabled
}

// InterruptionTimeout process
func InterruptionTimeout(duration time.Duration, interruption func(ClustStatus, error) error) (result Timeout) {
	result = &timeout{
		duration:     duration,
		enabled:      true,
		interruption: interruption,
	}
	go result.(*timeout).interrupt()
	return
}

// WaitTimeout process. Return true if timedout
func WaitTimeout(iter int, duration time.Duration, runtimeFiguresGetter func() (figures.RuntimeFigures, error), ack <-chan bool) error {
	var iterationsGetter = iterationsGetter(runtimeFiguresGetter)
	var maxIter = iterationsGetter() + iter
	if iter == 0 {
		maxIter = 0
	}
	var t = timeout{
		duration:         duration,
		ack:              ack,
		maxIter:          maxIter,
		iterationsGetter: iterationsGetter,
	}
	return t.wait()
}

func (timeout *timeout) interrupt() {
	time.Sleep(timeout.duration)
	if timeout.Enabled() {
		timeout.interruption(Failed, ErrTimeout)
	}
}

func (timeout *timeout) isElapsedIter() bool {
	return timeout.maxIter > 0 && timeout.maxIter <= timeout.iterationsGetter()
}

func (timeout *timeout) isAlive(lastTime time.Time) bool {
	return timeout.duration == 0 || time.Now().Before(lastTime)
}

func (timeout *timeout) wait() (err error) {
	err = ErrTimeout
	var step = 1 * time.Millisecond
	var elapsedTime = time.Now().Add(timeout.duration)
	for err == ErrTimeout && timeout.isAlive(elapsedTime) {
		select {
		case <-timeout.ack:
			err = nil
			break
		default:
			if timeout.isElapsedIter() {
				err = ErrElapsedIter
			}
		}
		time.Sleep(step)
	}
	return
}

func (timeout *timeout) Disable() {
	timeout.mutex.Lock()
	timeout.enabled = false
	timeout.mutex.Unlock()
}
