package algorithm

import (
	"fmt"
	"math"
)

type EdgeNode struct {
	Name string  `json:"name"`
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	//SortedPings []*FogNode
	//Pings       []float32
	Pings map[string]float32 `json:"pings"` //remove the map!!!
}

func (en *EdgeNode) Init() *EdgeNode {
	en.Pings = make(map[string]float32)
	return en
}

func (en *EdgeNode) Distance(other *FogNode) float64 {
	return math.Sqrt(math.Pow(en.X-other.X, 2) + math.Pow(en.Y-other.Y, 2))
}

func (en *EdgeNode) String() string {
	return fmt.Sprintf("X%f Y%f", en.X, en.Y)
}

type FogNode struct {
	X     float64            `json:"x"`
	Y     float64            `json:"y"`
	Name  string             `json:"name"`
	Pings map[string]float32 `json:"pings"`
	//	Active         bool
	//	ResourcesUsed  map[Resource]int
	//	ResourcesTotal map[Resource]int
}

func (fn *FogNode) Init() *FogNode {
	//	fn.ResourcesTotal = make(map[Resource]int)
	//	fn.ResourcesUsed = make(map[Resource]int)
	return fn
}

func (fn *FogNode) String() string {
	return fmt.Sprintf("X%d Y%d", fn.X, fn.Y)
}

/*func (fn *FogNode) IsFull() bool {
	full := false

	for res, amount := range fn.ResourcesUsed {
		if amount >= fn.ResourcesTotal[res] {
			full = true
		}
	}

	return full
}*/

/*func (fn *FogNode) IsUnderutilized() bool {
	underfull := true

	for res, amount := range fn.ResourcesUsed {
		if float64(amount) >= 0.2*float64(fn.ResourcesTotal[res]) {
			underfull = false
		}
	}

	return underfull
}*/

/*func (fn *EdgeNode) GetPing(fNode *FogNode) float32 {
	idx := 0
	var ping float32
	found := false
	for idx < len(fn.SortedPings) && !found {
		if fn.SortedPings[idx] == fNode {
			ping = fn.Pings[idx]
			found = true
		}

		idx++
	}
	return ping
}*/
