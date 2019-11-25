package figures

// Key figure name
type Key string

// Value figure value
type Value float64

// RuntimeFigures are meta values given by respective impl
type RuntimeFigures map[Key]Value

const (
	Iterations   = Key("iterations")
	Acceptations = Key("acceptations")
	MaxDistance  = Key("maxDistance")
)
