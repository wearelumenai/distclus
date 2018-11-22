package series

import "distclus/core"

// Conf defines series configuration
type Conf struct {
	InnerSpace string
	Window     int
	Space      core.Space
}
