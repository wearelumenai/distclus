package core

type Buffer interface {
	Push(elemt Elemt)
	SetAsync()
}

// DataBuffer that stores data.
// In synchronous mode, when pushed() is called data are stored.
// In asynchronous mode, when pushed() is called data are staged.
// Staged data are stored when apply() is called.
type DataBuffer struct {
	pipe     chan Elemt
	Data     []Elemt
	async    bool
	strategy bufferSizeStrategy
}

// Creates a fixed size buffer if given size > 0.
// Otherwise creates an infinite size buffer.
func NewDataBuffer(data []Elemt, size int) *DataBuffer {
	var buf = DataBuffer{
		pipe:  make(chan Elemt, 2000),
		async: false,
	}

	switch {
	case size > len(data):
		// fixed size buffer, less data than buffer size
		buf.strategy = &fixedSizeStrategy{size, len(data)}
		buf.Data = make([]Elemt, len(data), size)
		copy(buf.Data[:len(data)], data)

	case size > 0:
		// fixed size buffer, more data than buffer size
		buf.strategy = &fixedSizeStrategy{size, size}
		buf.Data = make([]Elemt, size)
		copy(buf.Data, data[len(data)-size:])

	default:
		// infinite buffer
		buf.strategy = &infiniteSizeStrategy{ }
		buf.Data = make([]Elemt, len(data))
		copy(buf.Data, data)
	}

	return &buf
}

// Stores or stages an element depending on synchronous / asynchronous mode.
func (b *DataBuffer) Push(elmt Elemt) {
	if b.async {
		b.pipe <- elmt
	} else {
		b.Data = b.strategy.push(b.Data, elmt)
	}
}

func (b *DataBuffer) SetAsync() {
	b.async = true
}

// Applies all staged data in asynchronous mode, otherwise do nothing
func (b *DataBuffer) Apply() {
	for loop := b.async; loop; {
		loop = b.apply_next()
	}
}

// Applies next staged data if available and returns true.
// Otherwise returns false.
func (b *DataBuffer) apply_next() bool {
	var loop = true

	select {
	case elmt, ok := <-b.pipe:
		if ok {
			b.Data = b.strategy.push(b.Data, elmt)
		}
		loop = ok
	default:
		loop = false
	}

	return loop
}

// Handle the way data are stored, i.e. infinite or fixed size buffer.
type bufferSizeStrategy interface {
	push(data []Elemt, elemt Elemt) []Elemt
}

// Fixed size buffer
type fixedSizeStrategy struct {
	size int
	position int
}

func (s *fixedSizeStrategy) push(data []Elemt, elemt Elemt) []Elemt {
	if s.position == s.size {
		s.position = 0
	}

	if s.position < len(data) {
		data[s.position] = elemt
	} else {
		data = append(data, elemt)
	}

	s.position += 1
	return data
}

// Infinite size buffer
type infiniteSizeStrategy struct {
}

func (s *infiniteSizeStrategy) push(data []Elemt, elemt Elemt) []Elemt {
	return append(data, elemt)
}

