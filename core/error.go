package core

import "errors"

// ErrNotStarted fired if centroids or runtimefigures asked without running the algorithm
var ErrNotStarted = errors.New("clustering not started")

// ErrNotRunning raised while algorithm status equals Created, Ready or Failed
var ErrNotRunning = errors.New("Algorithm is not running")

// ErrRunning raised while algorithm status equals Running, Idle or Sleeping
var ErrRunning = errors.New("Algorithm is running")

// ErrSleeping raised while algorithm status equals Sleeping
var ErrSleeping = errors.New("Algorithm is sleeping")

// ErrStopping raised while algorithm status equals stopping
var ErrStopping = errors.New("Algorithm is stopping")

// ErrNotIdle idle status is asked and not setted
var ErrNotIdle = errors.New("Algorithm is not idle")

// ErrIdle raised if algo is idle
var ErrIdle = errors.New("Algorithm is idle")

// ErrInfiniteIterations occure when static execution is asked with infinite iterations
var ErrInfiniteIterations = errors.New("Infinite iterations in static mode")

// ErrTimeOut is returned when an error occurs
var ErrTimeOut = errors.New("algorithm timed out")

// ErrNeverSleeping raised when sleeping methods is called while the algorithm will never sleep
var ErrNeverSleeping = errors.New("algorithm can not sleep. Specify core.Conf.Iter or core.Conf.DataPerIter for allowing your algorithm to sleep")
