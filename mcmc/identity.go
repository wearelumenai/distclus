package mcmc

import (
	"distclus/core"
)

type Identity struct {
}

func NewIdentity() Identity {
	return Identity{}
}

func (Identity) Sample(mu core.Elemt, time int) core.Elemt {
	return mu
}

func (Identity) Pdf(x, mu core.Elemt, time int) float64 {
	// this works because x == mu in all cases
	return 1
}
