package algo

import "distclus/core"

type Buffer struct {
	pipe  chan core.Elemt
	Data  []core.Elemt
	async bool
	fz    int
	ci    int
}

func newBuffer(data []core.Elemt, fz int) Buffer {
	var buf = Buffer{
		pipe:  make(chan core.Elemt, 2000),
		async: false,
		fz:    fz,
	}

	switch {
	case data != nil && fz > len(data):
		buf.Data = make([]core.Elemt, len(data), fz)
		copy(buf.Data[:len(data)], data)
		buf.ci = len(data)

	case data != nil && fz > 0:
		buf.Data = make([]core.Elemt, fz)
		copy(buf.Data, data[len(data)-fz:])
		buf.ci = fz

	case data != nil:
		buf.Data = make([]core.Elemt, len(data))
		copy(buf.Data, data)
		buf.ci = len(data)

	default:
		buf.Data = make([]core.Elemt, 0)
		buf.ci = 0
	}

	return buf
}

func (b *Buffer) push(elmt core.Elemt) {
	_push(b, elmt, b.async)
}

func _push(b *Buffer, elmt core.Elemt, async bool) {
	switch {
	case async:
		b.pipe <- elmt

	case b.fz < 0 || b.ci == len(b.Data) && b.ci < b.fz:
		b.Data = append(b.Data, elmt)
		b.ci += 1

	case b.ci == b.fz:
		b.Data[0] = elmt
		b.ci = 1

	default:
		b.Data[b.ci] = elmt
		b.ci += 1
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
				_push(b, elmt, false)
			}
			loop = ok
		default:
			loop = false
		}
	}
}
