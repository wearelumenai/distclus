package core

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[string]float64

const (
	// Iterations is the total number of iterations
	Iterations = "iterations"
	// MaxDistance is the max distance of streaming
	MaxDistance = "maxDistance"
	// PushedData is the number of pushed data
	PushedData = "pushedData"
	// LastIterations is the last number of iterations
	LastIterations = "lastIterations"
	// Duration is total algo duration
	Duration = "duration"
	// LastDuration is the last execution duration
	LastDuration = "lastDuration"
	// LastDataTime is the last pushed data time
	LastDataTime = "lastDataTime"
)
