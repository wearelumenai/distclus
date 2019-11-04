package figures

type Figure string
type Value float64

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[Figure]Value

const (
	Iterations   = "iterations"
	Acceptations = "acceptations"
	MaxDistance  = "maxDistance"
)
