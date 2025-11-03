package algorithm

import (
	"fmt"
	"image"
	"math"
	"math/rand"
	"oamswirly/generator/config"
	"os/exec"
	"sort"
)

var FogNodes []*FogNode
var EdgeNodes []*EdgeNode

var epPixels []float64
var fpPixels []float64
var pixels []image.Point

func ExecCmdBash(dfCmd string) (string, error) {
	cmd := exec.Command("sh", "-c", dfCmd)
	stdout, err := cmd.Output()

	if err != nil {
		println(err.Error())
		return "", err
	}
	return string(stdout), nil
}

// This function generates a number of fog nodes at random positions, and a number of "cores" of edge nodes representing "towns".
// More or less the same as the GetMemoryForClusters function above, but this one keeps the generated topology around for use in the ClusterIncremental...
// functions and DeleteNodes function below.
func GenerateNodes(densityMap image.Image, numEdge int, numFog int, pingDiff float64) {
	FogNodes = []*FogNode{}
	EdgeNodes = []*EdgeNode{}

	loadDensity(densityMap)

	//create fog
	for i := 0; i < numFog; i++ {
		chance := rand.Float64()

		pFound := binaryFind(fpPixels, chance)
		node := FogNode{
			X:    float64(pFound.X),
			Y:    float64(pFound.Y),
			Name: fmt.Sprintf("f%d", i),
		}
		FogNodes = append(FogNodes, &node)
	}

	//build fog distance maps

	maxRelDiff := pingDiff / 100
	pings := make([][]float32, numFog)
	for f := 0; f < numFog; f++ {
		pings[f] = make([]float32, numFog)
	}
	for from := 0; from < numFog; from++ {
		for to := 0; to < numFog; to++ {
			if from != to {
				dist := math.Sqrt(math.Pow(FogNodes[from].X-FogNodes[to].X, 2) + math.Pow(FogNodes[from].Y-FogNodes[to].Y, 2))
				ping := ((1 - maxRelDiff) + rand.Float64()*(2*maxRelDiff)) * dist / 20
				pings[from][to] = float32(ping)
				pings[to][from] = float32(ping)
			}
		}
	}

	for from := 0; from < numFog; from++ {

		fogNodePings := make(map[string]float32)
		for to := 0; to < numFog; to++ {
			if from != to {
				fogNodePings[FogNodes[to].Name] = pings[from][to]
			}
		}
		FogNodes[from].Pings = fogNodePings
	}

	//create edge and distance maps

	for i := 0; i < numEdge; i++ {
		chance := rand.Float64()

		pFound := binaryFind(epPixels, chance)

		fogNodePings := make(map[string]float32)

		for _, fn := range FogNodes {
			dist := math.Sqrt(math.Pow(fn.X-float64(pFound.X), 2) + math.Pow(fn.Y-float64(pFound.Y), 2))
			ping := ((1 - maxRelDiff) + rand.Float64()*(2*maxRelDiff)) * dist / 20
			fogNodePings[fn.Name] = float32(ping)
		}

		//nodes, pings := SortNodePings(fogNodePings)

		node := EdgeNode{
			X:     float64(pFound.X),
			Y:     float64(pFound.Y),
			Pings: fogNodePings, //pings,
			//SortedPings: nodes,
			Name: fmt.Sprintf("e%d", i),
		}
		EdgeNodes = append(EdgeNodes, &node)
	}

}

func binaryFind(pPixels []float64, chance float64) image.Point {
	pFound := image.Point{X: 0, Y: 0}
	found := false
	curIdx := int(len(pPixels) / 2)
	step := curIdx

	for !found {
		current := pPixels[curIdx]
		if current == chance {
			found = true
			pFound = pixels[curIdx]
		} else {
			step /= 2
			if step == 1 {
				found = true
				if chance > current {
					pFound = pixels[curIdx]
				} else {
					pFound = pixels[curIdx-1]
				}
			} else {
				if chance > current {
					curIdx += step
				} else {
					curIdx -= step
				}
			}
		}
	}
	return pFound
}

func loadDensity(dmap image.Image) {
	pDensities := [1200][750]float64{}
	total := float64(0)
	//max := 0

	//edge node densities
	//For edge nodes, only the population density matters, so we normalize the entire map so the sum of all chances = 1
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			r, g, b, _ := dmap.At(x, y).RGBA()
			r /= 256
			g /= 256
			b /= 256
			if b != 255 {
				pDensities[x][y] = float64(b*255*255 + g*255 + r)
				total += pDensities[x][y]
			}
		}
	}

	epPixels = []float64{}
	pixels = []image.Point{}
	pTotal := float64(0)
	//This loop normalizes the chances according to total summed in previous loop
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			pTotal += pDensities[x][y] / total
			epPixels = append(epPixels, pTotal)
			pixels = append(pixels, image.Point{X: x, Y: y})
		}
	}

	//fog node densities
	//This is a bitch. Have to take into account both a chance for a node to be place according to r_p and according to rho_e

	fpPixels = []float64{}

	//First, read in rho_e
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			r, g, b, _ := dmap.At(x, y).RGBA()
			r /= 256
			g /= 256
			b /= 256
			if b != 255 {
				pDensities[x][y] = float64(b*255*255 + g*255 + r)
			}
		}
	}

	maxPing := float32(config.Cfg.SLAMaxPing)
	//pingVar := 1 + (maxPing / 100)
	capacity := float64(50000) //50000 / (pingVar * pingVar))
	minDensity := float64(1 / (math.Pi * maxPing * maxPing))
	//The following should be considered a magic number. In the original tests, the area of a bitmap was around 30x22km, which gives around 25mx25m per pixel
	//Ergo, a single pixel's chance of containing a fog node is pop density / node capacity * physical area represented by a pixel
	//Also, this is in REAL WORLD population densities, not the one used in tests, so the capacity of a fog node has been bumped to 50000 to reflect that
	estPixelCapacity := 0.025 * 0.025 / capacity

	//Calculate the chance of placing a fog node according to rho_e, if the minimum density required to satisfy r_p is large, use that
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			_, _, b, _ := dmap.At(x, y).RGBA()
			b /= 256
			//if b != 255 {
			pDensities[x][y] *= estPixelCapacity
			pDensities[x][y] = math.Max(pDensities[x][y], minDensity)
			//}
		}
	}

	total = 0
	//Calculate the total in order to normalize
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			_, _, b, _ := dmap.At(x, y).RGBA()
			b /= 256
			//if b != 255 {
			total += pDensities[x][y]
			//}
		}
	}

	fmt.Printf("Fog only total densities %f\n", minDensity*1200*750)
	fmt.Printf("Combined total densities %f\n", total)

	//minFogNodes := 1 /

	pTotal = float64(0)
	//Bloody hell, almost done, just need to normalize the map so the sum of all values = 1
	for x := 0; x < dmap.Bounds().Dx() && x < 1200; x++ {
		for y := 0; y < dmap.Bounds().Dy() && y < 750; y++ {
			_, _, b, _ := dmap.At(x, y).RGBA()
			b /= 256
			if b != 255 {
				pTotal += pDensities[x][y] / total
				fpPixels = append(fpPixels, pTotal)
			}
		}
	}
}

// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Again, a horrible golang mess. Because golang provides no real way to sort maps according to their key values,
// this thing had to be written.
func SortNodePings(pings map[*FogNode]float32) ([]*FogNode, []float32) {
	nodePings := []NodePing{}
	for fn, ping := range pings {
		nPing := NodePing{Node: fn, Ping: ping}
		nodePings = append(nodePings, nPing)
	}

	ping := func(p1, p2 *NodePing) bool {
		return p1.Ping < p2.Ping
	}
	By(ping).Sort(nodePings)

	nodes := []*FogNode{}
	npings := []float32{}
	for i := 0; i < len(nodePings); i++ {
		nodes = append(nodes, nodePings[i].Node)
		npings = append(npings, nodePings[i].Ping)
	}
	return nodes, npings
}

type NodePing struct {
	Node *FogNode
	Ping float32
}

type By func(p1, p2 *NodePing) bool

// Sort is a method on the function type, By, that sorts the argument slice according to the function.
func (by By) Sort(pings []NodePing) {
	ps := &pingSorter{
		pings: pings,
		by:    by, // The Sort method's receiver is the function (closure) that defines the sort order.
	}
	sort.Sort(ps)
}

type pingSorter struct {
	pings []NodePing
	by    func(p1, p2 *NodePing) bool // Closure used in the Less method.
}

// Len is part of sort.Interface.
func (s *pingSorter) Len() int {
	return len(s.pings)
}

// Swap is part of sort.Interface.
func (s *pingSorter) Swap(i, j int) {
	s.pings[i], s.pings[j] = s.pings[j], s.pings[i]
}

// Less is part of sort.Interface. It is implemented by calling the "by" closure in the sorter.
func (s *pingSorter) Less(i, j int) bool {
	return s.by(&s.pings[i], &s.pings[j])
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

func SortENPings(pings map[float32]*EdgeNode) ([]*EdgeNode, []float32) {
	nodePings := []ENodePing{}
	for ping, fn := range pings {
		nPing := ENodePing{Node: fn, Ping: ping}
		nodePings = append(nodePings, nPing)
	}

	ping := func(p1, p2 *ENodePing) bool {
		return p1.Ping < p2.Ping
	}
	ENBy(ping).Sort(nodePings)

	nodes := []*EdgeNode{}
	npings := []float32{}
	for i := 0; i < len(nodePings); i++ {
		nodes = append(nodes, nodePings[i].Node)
		npings = append(npings, nodePings[i].Ping)
	}
	return nodes, npings
}

type ENodePing struct {
	Node *EdgeNode
	Ping float32
}

type ENBy func(p1, p2 *ENodePing) bool

// Sort is a method on the function type, By, that sorts the argument slice according to the function.
func (by ENBy) Sort(pings []ENodePing) {
	ps := &ePingSorter{
		pings: pings,
		by:    by, // The Sort method's receiver is the function (closure) that defines the sort order.
	}
	sort.Sort(ps)
}

type ePingSorter struct {
	pings []ENodePing
	by    func(p1, p2 *ENodePing) bool // Closure used in the Less method.
}

// Len is part of sort.Interface.
func (s *ePingSorter) Len() int {
	return len(s.pings)
}

// Swap is part of sort.Interface.
func (s *ePingSorter) Swap(i, j int) {
	s.pings[i], s.pings[j] = s.pings[j], s.pings[i]
}

// Less is part of sort.Interface. It is implemented by calling the "by" closure in the sorter.
func (s *ePingSorter) Less(i, j int) bool {
	return s.by(&s.pings[i], &s.pings[j])
}
