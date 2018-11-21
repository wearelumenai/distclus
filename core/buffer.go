package core

// Buffer interface
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

// Maximal default pipe size
const pipeSize = 2000

// NewDataBuffer creates a fixed size buffer if given size > 0.
// Otherwise creates an infinite size buffer.
func NewDataBuffer(data []Elemt, size int) *DataBuffer {
	var buf = DataBuffer{
		pipe:  make(chan Elemt, pipeSize),
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
		buf.strategy = &infiniteSizeStrategy{}
		buf.Data = make([]Elemt, len(data))
		copy(buf.Data, data)
	}

	return &buf
}

// Push stores or stages an element depending on synchronous / asynchronous mode.
func (b *DataBuffer) Push(elmt Elemt) {
	if b.async {
		b.pipe <- elmt
	} else {
		b.Data = b.strategy.push(b.Data, elmt)
	}
}

// SetAsync set asynchronous execution status to true
func (b *DataBuffer) SetAsync() {
	b.async = true
}

// Apply all staged data in asynchronous mode, otherwise do nothing
func (b *DataBuffer) Apply() {
	for loop := b.async; loop; {
		loop = b.applyNext()
	}
}

// Applies next staged data if available and returns true.
// Otherwise returns false.
func (b *DataBuffer) applyNext() (loop bool) {
	loop = true

	select {
	case elmt, ok := <-b.pipe:
		if ok {
			b.Data = b.strategy.push(b.Data, elmt)
		}
		loop = ok
	default:
		loop = false
	}

	return
}

// Handle the way data are stored, i.e. infinite or fixed size buffer.
type bufferSizeStrategy interface {
	push(data []Elemt, elemt Elemt) []Elemt
}

// Fixed size buffer
type fixedSizeStrategy struct {
	size     int
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

	s.position++

	return data
}

// Infinite size buffer
type infiniteSizeStrategy struct {
}

func (s *infiniteSizeStrategy) push(data []Elemt, elemt Elemt) []Elemt {
	return append(data, elemt)
}
