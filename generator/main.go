package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	_ "image/png"
	"math"
	"math/rand"
	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	discoverycfg "oamswirly/discoveryservice/config"
	"oamswirly/generator/algorithm"
	"oamswirly/generator/config"
	"os"

	log "github.com/sirupsen/logrus"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	//log.Infof("Loading config file %s", cfgFile)
	config.LoadConfig(cfgFile)

	//determine what type of test needs to be done. It's best to do only one of these at a time, but in theory they can all run in sequence.
	//if config.Cfg.SpeedTest {
	BuildConfigs()
	//}
}

// Rounds a number up to a multiple of FogNodeStep, required for finding the minimum number of fog nodes for an edge infrastructure and then
// making it adhere to the test settings
func RoundUp(number float64) int {
	steps := number / float64(config.Cfg.FogNodeStep)
	return config.Cfg.FogNodeStep * int(math.Ceil(steps))
}

// This builds a service topology and measures the timings of the add and delete operations.
// For automation purposes, it iterates from MinEdgeNodes to MaxEdgeNodes in EdgeNodeStep steps.
// The same goes for MinFogNodes, MaxFogNodes and FogNodeStep, however MinFogNodes can be overriden by the magic number 800 (see below).
// Because the generated fog nodes and edge nodes are completely random, any topology may turn out to be wildly positive or negative, skewing the results.
// Therefore, it is recommended to do a good number of runs per edgenode/fognode step (Iterations = 20 seems good).
// TODO: get rid of the magic number "800": it has to do with how many clients can "safely" fit on the average fog node given the hardcoded
// resource constraints in clustering.go.
// In other words, if we take more than 800, the algorithm will GIGO because it can't find a spot for every edge node, even a bad one.
func BuildConfigs() {
	//slaMaxPing := float32(config.Cfg.SLAMaxPing)
	maxPingDiff := float64(config.Cfg.MaxPingDiff)
	file, err := os.Open(config.Cfg.DensityMap)
	if err != nil {
		log.Errorf("Couldn't open %s %v", config.Cfg.DensityMap, err)
		return
	}
	defer file.Close()
	densityMapPng, _, err := image.Decode(file)
	hslDMap := hslDensityMap(densityMapPng)

	//iterate over number of edge nodes
	for en := config.Cfg.MinEdgeNodes; en <= config.Cfg.MaxEdgeNodes; en += config.Cfg.EdgeNodeStep {

		//iterate over number of fog nodes, start at a minimum of 1 per 800 edge nodes
		fnLimit := float64(RoundUp(float64(en) / 1000))
		for fn := int(math.Max(fnLimit, float64(config.Cfg.MinFogNodes))); fn <= config.Cfg.MaxFogNodes; fn += config.Cfg.FogNodeStep {
			//for iter := 0; iter < config.Cfg.Iterations; iter++ {
			algorithm.GenerateNodes(densityMapPng, en, fn, maxPingDiff)

			imgTopo := image.NewRGBA(hslDMap.Bounds())
			draw.Draw(imgTopo, imgTopo.Bounds(), hslDMap, image.Point{0, 0}, draw.Src)

			fogPath := fmt.Sprintf("fog/f%d-e%d", fn, en)
			edgePath := fmt.Sprintf("edge/f%d-e%d", fn, en)
			os.MkdirAll(fogPath, os.ModePerm)
			os.MkdirAll(edgePath, os.ModePerm)

			knownNodes := []nodediscovery.SvcNode{}

			/*for counter := 0; counter < len(algorithm.FogNodes); counter++ {
				fogNode := algorithm.FogNodes[counter]
				knownNodes[fogNode.Name] = nodediscovery.SvcNode{
					Name:         fogNode.Name,
					IP:           "127.0.0.1",
					DiscoAPIPort: 31000 + counter,
					RepoAPIPort:  32000 + counter,
				}
			}*/
			//staticKnownNodes := make(map[string]string)
			//staticKnownNodes["f0"] = "127.0.0.1"
			for counter := 0; counter < len(algorithm.FogNodes); counter++ { // _, fogNode := range algorithm.FogNodes {
				fogNode := algorithm.FogNodes[counter]
				cheatyPings := make(map[string]float32)
				for fn, ping := range fogNode.Pings {
					cheatyPings[fn] = ping
				}

				//cheatyEdgePings := make(map[string]float32)
				for _, en := range algorithm.EdgeNodes {
					cheatyPings[en.Name] = en.Pings[fogNode.Name]
				}

				fnCfg := discoverycfg.Config{
					DiscoAPIPort:         31000 + counter,
					RepoAPIPort:          32000 + counter,
					InitialNodes:         pickKnownNodes(knownNodes),
					MaxPing:              float32(config.Cfg.SLAMaxPing),
					NodeID:               fogNode.Name,
					NodeType:             string(nodediscovery.NodeTypeService),
					PingPeriod:           3000,
					TestMode:             true,
					CheatyMinimalPing:    10,
					CheatyMinimalPingMap: cheatyPings,
					BasicComponentDefs:   generateKnownComponents(5),
				}

				cfgBytes, err := json.Marshal(fnCfg)
				if err != nil {
					//panic(err)
					log.Errorf("Fog JSON marshal error %v", err)
				}
				err = os.WriteFile(fmt.Sprintf("%s/fog%d.json", fogPath, counter), cfgBytes, 0644)
				if err != nil {
					//panic(err)
					log.Errorf("Fog config write error %v", err.Error())
				}

				knownNodes = append(knownNodes, nodediscovery.SvcNode{
					Name:         fogNode.Name,
					IP:           "127.0.0.1",
					DiscoAPIPort: 31000 + counter,
					RepoAPIPort:  32000 + counter,
				})
				drawPoint(imgTopo, image.Point{X: int(fogNode.X), Y: int(fogNode.Y)}, 7, color.RGBA{255, 0, 0, 255})
				//draw lines for connectedness
				for _, otherFn := range algorithm.FogNodes {
					if fogNode.Pings[otherFn.Name] < fnCfg.MaxPing {
						drawLine(imgTopo, image.Point{X: int(fogNode.X), Y: int(fogNode.Y)}, image.Point{X: int(otherFn.X), Y: int(otherFn.Y)}, 7, color.RGBA{255, 0, 0, 255})
					}
				}
			}

			fogBytes, err := json.Marshal(algorithm.FogNodes)
			if err != nil {
				//panic(err)
				log.Errorf("General fog JSON marshal error %v", err)
			}
			err = os.WriteFile(fmt.Sprintf("%s/fog.json", fogPath), fogBytes, 0644)
			if err != nil {
				//panic(err)
				log.Errorf("General fog config error %v", err)
			}

			//services := make(map[string][]string)
			//services["monitorservice1"] = []string{"supportservice1"}
			/*for counter := 0; counter < len(algorithm.EdgeNodes); counter++ { //for _, edgeNode := range algorithm.EdgeNodes {
				edgeNode := algorithm.EdgeNodes[counter]

				closest := float32(100000)
				for _, node := range algorithm.FogNodes {
					dist := edgeNode.Pings[node.Name]
					if dist < closest {
						closest = dist
					}
				}

				enCfg := discoverycfg.Config{
					DiscoAPIPort:         31000 + counter + len(algorithm.FogNodes),
					RepoAPIPort:          32000 + counter + len(algorithm.FogNodes),
					InitialNodes:         pickKnownNodes(knownNodes),
					MaxPing:              float32(config.Cfg.SLAMaxPing),
					NodeID:               edgeNode.Name, //fmt.Sprintf("f%d", counter),
					NodeType:             string(nodediscovery.NodeTypeService),
					PingPeriod:           3000,
					TestMode:             true,
					CheatyMinimalPing:    float32(closest),
					CheatyMinimalPingMap: make(map[string]float32),
				}

				cfgBytes, err := json.Marshal(enCfg)
				if err != nil {
					//panic(err)
					log.Errorf("Edge JSON marshal error %v", err)
				}
				err = os.WriteFile(fmt.Sprintf("%s/edge%d.json", edgePath, counter), cfgBytes, 0644)
				if err != nil {
					//panic(err)
					log.Errorf("Edge config write error %v", err)
				}

				//knownNodes[edgeNode.Name] = "127.0.0.1"
				drawPoint(imgTopo, image.Point{X: int(edgeNode.X), Y: int(edgeNode.Y)}, 5, color.RGBA{0, 150, 0, 255})
			}*/

			/*edgeBytes, err := json.Marshal(algorithm.EdgeNodes)
			if err != nil {
				//panic(err)
				log.Errorf("General edge JSON marshal error %v", err)
			}
			err = os.WriteFile(fmt.Sprintf("%s/edge.json", edgePath), edgeBytes, 0644)
			if err != nil {
				//panic(err)
				log.Errorf("General edge config error %v", err)
			}*/

			f, err := os.Create(fmt.Sprintf("%s/topo.png", fogPath))
			err = png.Encode(bufio.NewWriter(f), imgTopo)
			f.Close()
			//}
		}
	}

}

func generateKnownComponents(amount int) map[string][]*oam.ComponentDef {
	components := make(map[string][]*oam.ComponentDef)

	labels := make(map[string]string)
	reqs := v1.ResourceList{}
	reqs[v1.ResourceCPU], _ = resource.ParseQuantity("100m")
	reqs[v1.ResourceMemory], _ = resource.ParseQuantity("100M")
	limits := v1.ResourceList{}
	limits[v1.ResourceCPU], _ = resource.ParseQuantity("500m")
	limits[v1.ResourceMemory], _ = resource.ParseQuantity("300M")

	for i := 0; i < amount; i++ {
		genericCName := fmt.Sprintf("%d", rand.Intn(config.Cfg.NumComponents)+1)
		componentName := fmt.Sprintf("Component %s", genericCName)
		concreteVersion := fmt.Sprintf("%d", rand.Intn(config.Cfg.ImplsPerComponent)+1)

		cDef := oam.ComponentDef{
			ApiVersion: "v1/beta",
			Kind:       "ComponentDefinition",
			Metadata: oam.Metadata{
				Name:   fmt.Sprintf("Implementation %s.%s", genericCName, concreteVersion),
				Labels: labels,
			},
			Spec: oam.ComponentDefSpec{
				Workload: oam.WorkloadTypeDescriptor{
					Type: string(oam.ContainerRuntime),
				},
				Schematic: oam.Schematic{
					BaseComponent: componentName,
					Definition: v1.Container{
						Name:            "TestApp",
						Image:           "gitlab.ilabt.imec.be:4567/fledge/benchmark/dash:1.0.0-capstan",
						ImagePullPolicy: v1.PullAlways,
						Command:         []string{"./dashserver.so"},
						Resources: v1.ResourceRequirements{
							Requests: reqs,
							Limits:   limits,
						},
					},
				},
			},
		}
		_, exists := components[componentName]
		if exists {
			cdefs := append(components[componentName], &cDef)
			components[componentName] = cdefs
		} else {
			components[componentName] = []*oam.ComponentDef{&cDef}
		}
	}
	return components
}

func pickKnownNodes(list []nodediscovery.SvcNode) []nodediscovery.SvcNode {
	if len(list) < 5 {
		return list
	}

	nodes := []nodediscovery.SvcNode{}
	for i := 0; i < 5; i++ {
		idx := rand.Int31n(int32(len(list)))

		//count := int32(0)
		//key := ""
		//value := ""
		/*for key, value := range list {

			if count == idx {
				nodes[key] = value
			}
			count++
		}*/
		//delete(list, key)
		nodes = append(nodes, list[idx])
	}
	return nodes
}

func drawPoint(img *image.RGBA, p image.Point, size int, c color.Color) {
	for x := 0; x < size; x++ {
		for y := 0; y < size; y++ {
			img.Set(p.X+x, p.Y+y, c)
		}
	}
}

func drawLine(img *image.RGBA, p1 image.Point, p2 image.Point, size int, c color.Color) {
	diffX := p2.X - p1.X
	diffY := p2.Y - p1.Y

	absdiffX := int(math.Abs(float64(diffX)))
	absdiffY := int(math.Abs(float64(diffY)))

	if absdiffX > absdiffY {
		startP := p2
		//stopP := p1
		if p1.X < p2.X {
			startP = p1
			//stopP = p2
		}
		dY := float64(diffY) / float64(diffX)
		for x := 1; x < diffX; x++ {
			img.Set(startP.X+x, startP.Y+int(float64(x)*dY), c)
		}
	} else {
		startP := p2
		if p1.Y < p2.Y {
			startP = p1
		}
		dX := float64(diffX) / float64(diffY)
		for y := 1; y < diffY; y++ {
			img.Set(startP.X+int(float64(y)*dX), startP.Y+y, c)
		}
	}
}

func hslDensityMap(img image.Image) *image.RGBA {
	imgTopo := image.NewRGBA(img.Bounds())

	maxD := float64(0)
	for x := 0; x < img.Bounds().Dx(); x++ {
		for y := 0; y < img.Bounds().Dy(); y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r /= 256
			g /= 256
			b /= 256
			if b != 255 {
				density := float64(b*255*255 + g*255 + r)
				if density > maxD {
					maxD = density
				}
			}
		}
	}
	gradient := maxD / 255

	for x := 0; x < img.Bounds().Dx(); x++ {
		for y := 0; y < img.Bounds().Dy(); y++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r /= 256
			g /= 256
			b /= 256
			if b != 255 {
				density := float64(b*255*255 + g*255 + r)

				pValue := density / gradient * 360 / 255
				rv, gv, bv := hsvToRgb(180+pValue, 0.5, 1)
				imgTopo.Set(x, y, color.RGBA{rv, gv, bv, 255})
			}
		}
	}
	return imgTopo
}

func hsvToRgb(hue float64, S float64, V float64) (uint8, uint8, uint8) {
	H := hue
	for H < 0 {
		H += 360
	}
	for H >= 360 {
		H -= 360
	}

	R := float64(0)
	G := float64(0)
	B := float64(0)
	if V <= 0 {

	} else if S <= 0 {
		R = V
		G = V
		B = V
	} else {
		hf := H / 60.0
		i := int(math.Floor(hf))
		f := hf - float64(i)
		pv := V * (1 - S)
		qv := V * (1 - S*f)
		tv := V * (1 - S*(1-f))
		switch i {

		// Red is the dominant color

		case 0:
			R = V
			G = tv
			B = pv
			break

		// Green is the dominant color

		case 1:
			R = qv
			G = V
			B = pv
			break
		case 2:
			R = pv
			G = V
			B = tv
			break

		// Blue is the dominant color

		case 3:
			R = pv
			G = qv
			B = V
			break
		case 4:
			R = tv
			G = pv
			B = V
			break

		// Red is the dominant color

		case 5:
			R = V
			G = pv
			B = qv
			break

		// Just in case we overshoot on our math by a little, we put these here. Since its a switch it won't slow us down at all to put these here.

		case 6:
			R = V
			G = tv
			B = pv
			break
		case -1:
			R = V
			G = pv
			B = qv
			break

		// The color is not defined, we should throw an error.

		default:
			//LFATAL("i Value error in Pixel conversion, Value is %d", i);
			//R = G = B = V; // Just pretend its black/white
			R = V
			G = V
			B = V
			break
		}
	}
	r := clamp((int)(R * 255.0))
	g := clamp((int)(G * 255.0))
	b := clamp((int)(B * 255.0))
	return r, g, b
}

func clamp(i int) uint8 {
	if i < 0 {
		return 0
	}
	if i > 255 {
		return 255
	}
	return uint8(i)
}
