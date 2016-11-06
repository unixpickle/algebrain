package algebrain

import "reflect"

// splitReadWrite splits a slice into those elements
// corresponding to a read operation and those
// corresponding to a write operation.
func splitReadWrite(isReading []bool, slice, outRead, outWrite interface{}) {
	sv := reflect.ValueOf(slice)

	var readCount, writeCount int
	for _, x := range isReading {
		if x {
			readCount++
		} else {
			writeCount++
		}
	}

	readSlice := reflect.MakeSlice(sv.Type(), readCount, readCount)
	writeSlice := reflect.MakeSlice(sv.Type(), writeCount, writeCount)

	var readIdx, writeIdx int
	for i, x := range isReading {
		if x {
			readSlice.Index(readIdx).Set(sv.Index(i))
			readIdx++
		} else {
			writeSlice.Index(writeIdx).Set(sv.Index(i))
			writeIdx++
		}
	}

	reflect.ValueOf(outRead).Elem().Set(readSlice)
	reflect.ValueOf(outWrite).Elem().Set(writeSlice)
}

// joinReadWrite inverses the work of splitReadWrite.
func joinReadWrite(isReading []bool, readSlice, writeSlice, outSlice interface{}) {
	readS := reflect.ValueOf(readSlice)
	writeS := reflect.ValueOf(writeSlice)
	total := readS.Len() + writeS.Len()
	out := reflect.MakeSlice(readS.Type(), total, total)

	var readIdx, writeIdx int
	for i, x := range isReading {
		if x {
			out.Index(i).Set(readS.Index(readIdx))
			readIdx++
		} else {
			out.Index(i).Set(writeS.Index(writeIdx))
			writeIdx++
		}
	}
}
