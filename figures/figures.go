package figures

// Key figure name
type Key string

// Value figure value
type Value float64

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[Key]Value

const (
	// Iterations is the number of iterations
	Iterations = Key("iterations")
	// Acceptations is the number of acceptations of mcmc
	Acceptations = Key("acceptations")
	// MaxDistance is the max distance of streaming
	MaxDistance = Key("maxDistance")
	// PushedData is the number of pushed data
	PushedData = Key("pushedData")
)
