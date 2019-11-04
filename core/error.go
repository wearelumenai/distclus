package core

import "errors"

// ErrNotStarted fired if centroids or runtimefigures asked without running the algorithm
var ErrNotStarted = errors.New("clustering not started")

// ErrEnded fired when running operation done after closing the algorithm
var ErrEnded = errors.New("clustering ended")

// ErrNotRunning idle status is asked and not setted
var ErrNotRunning = errors.New("Algorithm is not running")

// ErrNotIdle idle status is asked and not setted
var ErrNotIdle = errors.New("Algorithm is not idle")

// ErrNotReady ready status is asked and not setted
var ErrNotReady = errors.New("Algorithm is not ready")

// ErrInfiniteIterations occure when static execution is asked with infinite iterations
var ErrInfiniteIterations = errors.New("Infinite iterations in static mode")

// ErrTimeOut is returned when an error occurs
var ErrTimeOut = errors.New("algorithm timed out")
