package figures

// Key figure name
type Key string

// Value figure value
type Value float64

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[Key]Value

const (
	Iterations   = "iterations"
	Acceptations = "acceptations"
	MaxDistance  = "maxDistance"
)
