package core

import (
	"log"
	"sync"
	"time"
)

// Timeout interface for managing timeout algorithm
type Timeout interface {
	Disable()
	Enabled() bool
}

// timeout algorithm timeout structure
type timeout struct {
	duration     time.Duration
	enabled      bool
	interruption func(error) error
	mutex        sync.RWMutex
	finishing    Finishing
	ocm          OCModel
}

func (t *timeout) Enabled() bool {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	return t.enabled
}

// InterruptionTimeout process
func InterruptionTimeout(duration time.Duration, interruption func(error) error) (result Timeout) {
	result = &timeout{
		duration:     duration,
		enabled:      true,
		interruption: interruption,
	}
	go result.(*timeout).interrupt()
	return
}

// WaitTimeout process. Return true if timedout
func WaitTimeout(finishing Finishing, duration time.Duration, ocm OCModel) error {
	if finishing == nil {
		finishing = NewStatusFinishing(true, Ready, Idle, Finished)
	}
	var t = timeout{
		duration:  duration,
		ocm:       ocm,
		finishing: finishing,
	}
	return t.wait()
}

func (t *timeout) interrupt() {
	time.Sleep(t.duration)
	if t.Enabled() {
		t.interruption(ErrTimeout)
	}
}

func (t *timeout) isFinished() bool {
	return IsFinished(t.finishing, t.ocm)
}

func (t *timeout) isTimeout(lastTime time.Time) bool {
	return t.duration > 0 && time.Now().After(lastTime)
}

func (t *timeout) wait() (err error) {
	var step = 100 * time.Millisecond
	var elapsedTime = time.Now().Add(t.duration)

	for {
		log.Println(t.ocm.RuntimeFigures())
		if t.isFinished() {
			break
		} else if t.isTimeout(elapsedTime) {
			err = ErrTimeout
			break
		} else {
			time.Sleep(step)
		}
	}

	if err == nil {
		err = t.ocm.Status().Error
	}
	return
}

func (t *timeout) Disable() {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	t.enabled = false
}
