package algo

import "distclus/core"

type Buffer struct {
	pipe  chan core.Elemt
	Data  []core.Elemt
	async bool
}

func newBuffer(data []core.Elemt) Buffer {
	return Buffer{
		pipe:  make(chan core.Elemt, 2000),
		Data:  data,
		async: false,
	}
}

func (b *Buffer) push(elmt core.Elemt) {
	if b.async {
		b.pipe <- elmt
	} else {
		b.Data = append(b.Data, elmt)
	}
}

func (b *Buffer) setAsync() {
	b.async = true
}

func (b *Buffer) apply() {
	for loop := b.async; loop; {
		select {
		case elmt, ok := <-b.pipe:
			if ok {
				b.Data = append(b.Data, elmt)
			}
			loop = ok
		default:
			loop = false
		}
	}
}
