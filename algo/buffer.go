package algo

import "distclus/core"

// Buffer that stores data.
// In synchronous mode, when pushed() is called data are stored.
// In asynchronous mode, when pushed() is called data are staged.
// Staged data are stored when apply() is called.
type Buffer struct {
	pipe     chan core.Elemt
	Data     []core.Elemt
	async    bool
	support  bufferSupport
}

// Handle the way data are stored, i.e. infinite or fixed size buffer.
type bufferSupport interface {
	push(b *Buffer, elemt core.Elemt)
}

// Creates a fixed size buffer if given size > 0.
// Otherwise creates an infinite size buffer.
func newBuffer(data []core.Elemt, size int) Buffer {
	var buf = Buffer{
		pipe:  make(chan core.Elemt, 2000),
		async: false,
	}

	switch {
	case size > len(data):
		// fixed size buffer, less data than buffer size
		buf.support = &fixedBufferSupport {size, len(data)}
		buf.Data = make([]core.Elemt, len(data), size)
		copy(buf.Data[:len(data)], data)

	case size > 0:
		// fixed size buffer, more data than buffer size
		buf.support = &fixedBufferSupport {size, size}
		buf.Data = make([]core.Elemt, size)
		copy(buf.Data, data[len(data)-size:])

	default:
		// infinite buffer
		buf.support = &infiniteBufferSupport { }
		buf.Data = make([]core.Elemt, len(data))
		copy(buf.Data, data)
	}

	return buf
}

// Stores or stages an element depending on synchronous / asynchronous mode.
func (b *Buffer) push(elmt core.Elemt) {
	if b.async {
		b.pipe <- elmt
	} else {
		b.support.push(b, elmt)
	}
}

func (b *Buffer) setAsync() {
	b.async = true
}

// Applies all staged data in asynchronous mode, otherwise do nothing
func (b *Buffer) apply() {
	for loop := b.async; loop; {
		loop = b.apply_next()
	}
}

// Applies next staged data if available and returns true.
// Otherwise returns false.
func (b *Buffer) apply_next() bool {
	var loop = true

	select {
	case elmt, ok := <-b.pipe:
		if ok {
			b.support.push(b, elmt)
		}
		loop = ok
	default:
		loop = false
	}

	return loop
}

// Fixed size buffer
type fixedBufferSupport struct {
	size int
	position int
}

func (s *fixedBufferSupport) push(b *Buffer, elemt core.Elemt) {
	if s.position == s.size {
		s.position = 0
	}

	if s.position < len(b.Data) {
		b.Data[s.position] = elemt
	} else {
		b.Data = append(b.Data, elemt)
	}

	s.position += 1
}

// Infinite size buffer
type infiniteBufferSupport struct {
}

func (s *infiniteBufferSupport) push(b *Buffer, elemt core.Elemt) {
	b.Data = append(b.Data, elemt)
}

