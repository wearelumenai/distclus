package core

// Buffer interface
type Buffer interface {
	Push(elemt Elemt) error
	SetAsync() error
	Data() []Elemt
	Apply() error
}

// DataBuffer that stores data.
// In synchronous mode, when pushed() is called data are stored.
// In asynchronous mode, when pushed() is called data are staged.
// Staged data are stored when apply() is called.
type DataBuffer struct {
	pipe     chan Elemt
	data     []Elemt
	async    bool
	strategy bufferSizeStrategy
}

// Maximal default pipe size
const pipeSize = 2000

// NewDataBuffer creates a fixed size buffer if given size > 0.
// Otherwise creates an infinite size buffer.
func NewDataBuffer(data []Elemt, size int) Buffer {
	var db = DataBuffer{
		pipe:  make(chan Elemt, pipeSize),
		async: false,
	}

	switch {
	case size > len(data):
		// fixed size buffer, less data than buffer size
		db.strategy = &fixedSizeStrategy{size, len(data)}
		db.data = make([]Elemt, len(data), size)
		copy(db.data[:len(data)], data)

	case size > 0:
		// fixed size buffer, more data than buffer size
		db.strategy = &fixedSizeStrategy{size, size}
		db.data = make([]Elemt, size)
		copy(db.data, data[len(data)-size:])

	default:
		// infinite buffer
		db.strategy = &infiniteSizeStrategy{}
		db.data = make([]Elemt, len(data))
		copy(db.data, data)
	}

	return &db
}

// Push stores or stages an element depending on synchronous / asynchronous mode.
func (b *DataBuffer) Push(elmt Elemt) (err error) {
	if b.async {
		b.pipe <- elmt
	} else {
		b.data = b.strategy.push(b.data, elmt)
	}
	return
}

// Data returns buffer data
func (b *DataBuffer) Data() (data []Elemt) {
	return b.data
}

// SetAsync set asynchronous execution status to true
func (b *DataBuffer) SetAsync() (err error) {
	b.async = true
	return
}

// Apply all staged data in asynchronous mode, otherwise do nothing
func (b *DataBuffer) Apply() (err error) {
	for loop := b.async; loop; {
		loop = b.applyNext()
	}
	return
}

// Applies next staged data if available and returns true.
// Otherwise returns false.
func (b *DataBuffer) applyNext() (loop bool) {
	loop = true

	select {
	case elmt, ok := <-b.pipe:
		if ok {
			b.data = b.strategy.push(b.data, elmt)
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
