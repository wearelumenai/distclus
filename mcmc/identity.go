package mcmc

import (
	"distclus/core"
	"reflect"
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
	if reflect.DeepEqual(x, mu) {
		return 1
	}
	return 0
}
