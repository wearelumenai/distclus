package figures

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[string]float64

const (
	// Iterations is the number of iterations
	Iterations = "iterations"
	// Acceptations is the number of acceptations of mcmc
	Acceptations = "acceptations"
	// MaxDistance is the max distance of streaming
	MaxDistance = "maxDistance"
	// PushedData is the number of pushed data
	PushedData = "pushedData"
)
