package main

import (
	"encoding/json"
	"fmt"
	_ "image/png"
	"math"
	"net/http"
	discovery "oamswirly/common/nodediscovery"
	"oamswirly/runner/config"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var fileFDs []*os.File

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	fileFDs = []*os.File{}
	config.LoadConfig(cfgFile)

	for en := config.Cfg.MinEdgeNodes; en <= config.Cfg.MaxEdgeNodes; en += config.Cfg.EdgeNodeStep {

		//iterate over number of fog nodes, start at a minimum of 1 per 800 edge nodes
		fnLimit := float64(RoundUp(float64(en) / 1000))
		for fn := int(math.Max(fnLimit, float64(config.Cfg.MinFogNodes))); fn <= config.Cfg.MaxFogNodes; fn += config.Cfg.FogNodeStep {
			for iter := 0; iter < config.Cfg.Iterations; iter++ {
				folderPath := fmt.Sprintf("f%d-e%d", fn, en)
				fmt.Printf("Fog %d edge %d iteration %d, using files in %s", fn, en, iter, folderPath)

				os.MkdirAll(folderPath, os.ModePerm)

				fogPids, edgePids := startNodes(fn, en, folderPath)
				monitorNodes(fogPids, edgePids)
				stopNodes(fogPids, edgePids)
			}
		}
	}
}

func RoundUp(number float64) int {
	steps := number / float64(config.Cfg.FogNodeStep)
	return config.Cfg.FogNodeStep * int(math.Ceil(steps))
}

func startNodes(numFogs int, numEdges int, prefix string) (map[string]int, map[string]int) {
	//numFogs := config.Cfg.NumFogNodes
	//numEdges := config.Cfg.NumEdgeNodes

	fogPids := make(map[string]int)
	edgePids := make(map[string]int)

	if numFogs > 1 {
		for nr := 0; nr < numFogs; nr++ {
			cfgFile := fmt.Sprintf("%s/fog%d.json", prefix, nr)

			fmt.Printf("Starting fog node from %s\n", cfgFile)
			//cmd := fmt.Sprintf("./run.sh  %s &", cfgFile)
			pid, err := startNode(cfgFile, prefix)
			if err != nil {
				fmt.Printf("Failed to start fog node %d: %s\n", nr, err.Error())
			} else {
				//pid, _ := strconv.Atoi(output)
				fogPids[fmt.Sprintf("f%d", nr)] = pid
				fmt.Printf("Started process %d for node\n", pid)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}

	if numEdges > 1 {
		for nr := 0; nr < numEdges; nr++ {
			cfgFile := fmt.Sprintf("%s/edge%d.json", prefix, nr)

			fmt.Printf("Starting edge node from %s\n", cfgFile)
			//cmd := fmt.Sprintf("./soswirlyedge %s &", cfgFile)
			//output, err := execCmdBash(cmd)
			pid, err := startEdgeNode(cfgFile, prefix)
			if err != nil {
				fmt.Printf("Failed to start edge node %d: %s\n", nr, err.Error())
			} else {
				//pid, _ := strconv.Atoi(output)
				edgePids[fmt.Sprintf("f%d", nr)] = pid
				fmt.Printf("Started process %d for node\n", pid)
			}
			time.Sleep(200 * time.Millisecond)
		}
		time.Sleep(60 * time.Second)
	}
	return fogPids, edgePids
}

func monitorNodes(fogPids map[string]int, edgePids map[string]int) {
	fogData := make(map[string][]StatsLine)
	edgeData := make(map[string][]StatsLine)
	generalData := []StatsLine{}

	for node, _ := range fogPids {
		fogData[node] = []StatsLine{}
	}

	for node, _ := range edgePids {
		edgeData[node] = []StatsLine{}
	}

	fmetricNames := []string{"memory", "cpu"} //, "neighbours", "accuracy", "inrange", "tolerated", "outsiderange"}
	emetricNames := []string{"memory", "cpu", "minping", "ping"}
	prevCpu := make(map[string]int)
	prevNetTraffic := 0
	prevLoNetTraffic := 0
	timeTaken := int64(0)
	for iteration := 0; iteration < config.Cfg.MonitorLoops; iteration++ {
		fmt.Printf("Monitor iteration %d, previous time taken %d\n", iteration, timeTaken)
		time.Sleep(time.Duration(config.Cfg.MonitorPeriod*1000-timeTaken) * time.Millisecond)
		startTime := time.Now()

		totalDiscovered := 0
		totalExpected := 0

		for node, pid := range fogPids {
			metrics := make(map[string]float64)
			metrics["memory"] = float64(getMemory(pid))

			prevNCpu, found := prevCpu[node]
			cpuvalue := float64(getCPU(pid))
			if !found {
				metrics["cpu"] = 0
			} else {
				metrics["cpu"] = (cpuvalue - float64(prevNCpu)) / float64(config.Cfg.MonitorPeriod)
			}
			prevCpu[node] = int(cpuvalue)

			//neighbours := getKnownFogNodes(node)
			//stats := getDiscoveredNodeStats(node)
			//neighbours := float64(stats.NodesWithinRange + stats.NodesInAcceptableRange + stats.OutsideRange)
			metrics["neighbours"] = 1 //neighbours

			//acc := math.Min(float64(stats.Discovered)/math.Max(1, float64(stats.ExpectedInRange)), 1)

			metrics["accuracy"] = 1     // acc
			metrics["inrange"] = 1      //float64(stats.NodesWithinRange) / neighbours
			metrics["tolerated"] = 0    //float64(stats.NodesInAcceptableRange) / neighbours
			metrics["outsiderange"] = 0 //float64(stats.OutsideRange) / neighbours

			totalDiscovered += 1 // stats.Discovered
			totalExpected += 1   //stats.ExpectedInRange

			line := StatsLine{
				MNames:  fmetricNames,
				Metrics: metrics,
			}
			fogData[node] = append(fogData[node], line)

		}

		serviced := 0
		fogservers := make(map[string]bool)
		for node, pid := range edgePids {
			metrics := make(map[string]float64)
			metrics["memory"] = float64(getMemory(pid))

			prevNCpu, found := prevCpu[node]
			cpuvalue := float64(getCPU(pid))
			if !found {
				metrics["cpu"] = 0
			} else {
				metrics["cpu"] = cpuvalue - float64(prevNCpu)
			}
			prevCpu[node] = int(cpuvalue)

			//stats := getNodeStats(node)

			/*metrics["minping"] = float64(stats.MinimalPing)
			metrics["ping"] = float64(stats.CurrentClosestPing)
			if stats.CurrentClosestPing > 0 {
				serviced++
				fogservers[stats.CurrentFogNode] = true
			}*/

			line := StatsLine{
				MNames:  emetricNames,
				Metrics: metrics,
			}
			edgeData[node] = append(edgeData[node], line)
		}

		totNetTraffic := float64(0)
		gMetrics := make(map[string]float64)
		lonetTraffic := getNetworkTraffic("lo")
		if prevLoNetTraffic > 0 {
			totNetTraffic = float64(lonetTraffic - prevLoNetTraffic)
		}
		prevLoNetTraffic = (lonetTraffic)

		netTraffic := getNetworkTraffic(config.Cfg.EthInterface)
		if prevNetTraffic > 0 {
			totNetTraffic += float64(netTraffic - prevNetTraffic)
		}
		prevNetTraffic = (netTraffic)

		gMetrics["network"] = totNetTraffic
		gMetrics["cpu"] = float64(getTotalCPU())
		gMetrics["discovered"] = float64(totalDiscovered)
		gMetrics["expected"] = float64(totalExpected)
		gMetrics["serviced"] = float64(serviced)
		gMetrics["fognodes"] = float64(len(fogservers))

		generalLine := StatsLine{
			MNames:  []string{"network", "cpu", "discovered", "expected", "serviced", "fognodes"},
			Metrics: gMetrics,
		}
		generalData = append(generalData, generalLine)
		timeTaken = time.Now().UnixMilli() - startTime.UnixMilli()
	}

	//printHeader := true
	//for _, lines := range fogData {
	/*if printHeader {
		fmt.Println(lines[0].LineHeader())
		printHeader = false
	}*/

	/*fmt.Println(node)
	for i := 0; i < len(lines); i++ {
		data, _ := lines[i].String()
		fmt.Println(data)
	}
	fmt.Println("")*/
	//}

	if len(fogData) > 0 {
		fogTimeGroups := []GroupStats{}
		for i := 0; i < config.Cfg.MonitorLoops; i++ {
			groupLines := []StatsLine{}
			for _, nodelines := range fogData {
				if nodelines[i].Metrics["accuracy"] > 0 {
					groupLines = append(groupLines, nodelines[i])
				}
				//fmt.Println(data)
			}
			fogTimeGroups = append(fogTimeGroups, MakeGroupStats(groupLines))
		}

		fmt.Println(fogTimeGroups[0].GroupHeader())
		for i := 0; i < len(fogTimeGroups); i++ {
			data, _ := fogTimeGroups[i].String()
			fmt.Println(data)
		}
		fmt.Println("")
	}

	if len(edgeData) > 0 {
		edgeTimeGroups := []GroupStats{}
		for i := 0; i < config.Cfg.MonitorLoops; i++ {
			groupLines := []StatsLine{}
			for _, nodelines := range edgeData {
				if nodelines[i].Metrics["ping"] > 0 {
					groupLines = append(groupLines, nodelines[i])
				}
			}
			edgeTimeGroups = append(edgeTimeGroups, MakeGroupStats(groupLines))
		}

		fmt.Println(edgeTimeGroups[0].GroupHeader())
		for i := 0; i < len(edgeTimeGroups); i++ {
			data, _ := edgeTimeGroups[i].String()
			fmt.Println(data)
		}
		fmt.Println("")
	}

	fmt.Println(generalData[0].LineHeader())
	for i := 0; i < len(generalData); i++ {
		data, _ := generalData[i].String()
		fmt.Println(data)
	}
	fmt.Println("")
}

func stopNodes(fogPids map[string]int, edgePids map[string]int) {
	for _, pid := range fogPids {
		execCmdBash(fmt.Sprintf("kill %d", pid))
	}

	for _, pid := range edgePids {
		execCmdBash(fmt.Sprintf("kill %d", pid))
	}
}

func startNode(cfg string, prefix string) (int, error) {
	cmd := exec.Command("./discoverysvc", cfg)
	//fmt.Println(cmd)

	parts := strings.Split(cfg, "/")
	fname := parts[len(parts)-1]
	outfilename := fmt.Sprintf("%s/out%s.txt", prefix, fname)
	fmt.Printf("Logging node output to %s\n", outfilename)
	outfile, _ := os.Create(outfilename)
	fileFDs = append(fileFDs, outfile)
	cmd.Stdout = outfile
	cmd.Stderr = outfile

	err := cmd.Start()

	if err != nil {
		println(err.Error())
		return 0, err
	}
	return cmd.Process.Pid, nil
}

func startEdgeNode(cfg string, prefix string) (int, error) {
	cmd := exec.Command("./soswirlyedge", cfg)
	//fmt.Println(cmd)

	parts := strings.Split(cfg, "/")
	fname := parts[len(parts)-1]
	outfilename := fmt.Sprintf("%s/out%s.txt", prefix, fname)
	//fmt.Printf("Logging node output to %s\n", outfilename)
	outfile, _ := os.Create(outfilename)
	cmd.Stdout = outfile

	err := cmd.Start()

	if err != nil {
		println(err.Error())
		return 0, err
	}
	return cmd.Process.Pid, nil
}

func execCmdBash(dfCmd string) (string, error) {
	//fCmd := fmt.Sprintf("\"%s\"", dfCmd)
	cmd := exec.Command("sh", "-c", dfCmd)
	//fmt.Println(cmd)
	stdout, err := cmd.Output()

	if err != nil {
		println(err.Error())
		return "", err
	}
	return string(stdout), nil
}

func getMemory(pid int) int {
	file := fmt.Sprintf("/proc/%d/stat", pid)
	output, _ := execCmdBash(fmt.Sprintf("cat %s", file))
	parts := strings.Split(output, " ")

	mem, _ := strconv.Atoi(parts[23])
	return mem * 4
}

var reInsideWhtsp = regexp.MustCompile(`\s+`)

func getCPU(pid int) int {
	file := fmt.Sprintf("/proc/%d/stat", pid)
	output, _ := execCmdBash(fmt.Sprintf("cat %s", file))
	parts := strings.Split(reInsideWhtsp.ReplaceAllString(output, " "), " ")

	usercpu, _ := strconv.Atoi(parts[13])
	syscpu, _ := strconv.Atoi(parts[14])
	return (usercpu + syscpu) //should be /100 but if we want it in % then leave it like this
}

func getNetworkTraffic(itf string) int {
	cmd := fmt.Sprintf("cat /proc/net/dev | grep %s", itf)
	stats, _ := execCmdBash(cmd)
	parts := strings.Split(reInsideWhtsp.ReplaceAllString(stats, " "), " ")
	sent, _ := strconv.Atoi(parts[2])
	received, _ := strconv.Atoi(parts[10])
	return sent + received
}

func getCores() int {
	stdout, _ := execCmdBash("nproc")
	numCpus := strings.Trim(string(stdout), "\n")
	//fmt.Println(numCpus)
	cpusInt, _ := strconv.Atoi(numCpus)
	return cpusInt
}

func getTotalCPU() int {
	iostatc, _ := execCmdBash("iostat -c 1 2")
	var cpuUsed string
	var cpuLines = strings.Split(iostatc, "\n")
	//fmt.Println(memFree)
	cpuUsed = strings.Split(reInsideWhtsp.ReplaceAllString(cpuLines[8], " "), " ")[6]

	cpuPct, _ := strconv.ParseFloat(cpuUsed, 64)
	return int((100 - cpuPct) * float64(getCores()))
}

func getDiscoveredNodeStats(node string) discovery.DiscoveredNodes {
	nodeNr, _ := strconv.Atoi(node[1:])
	port := config.Cfg.DiscoAPIPort + nodeNr
	url := fmt.Sprintf("http://127.0.0.1:%d/%s", port, "getDiscoveredNodeStats")

	response, err := http.Get(url)

	if err != nil {
		fmt.Println(err.Error())
		return discovery.DiscoveredNodes{}
	}

	fogNodes := discovery.DiscoveredNodes{}
	err = json.NewDecoder(response.Body).Decode(&fogNodes)
	if err != nil {
		fmt.Println(err.Error())
		return discovery.DiscoveredNodes{}
	}
	response.Body.Close()
	return fogNodes
}

/*func getNodeStats(node string) discovery.NodeStats {
	nodeNr, _ := strconv.Atoi(node[1:])
	port := config.Cfg. + nodeNr
	url := fmt.Sprintf("http://127.0.0.1:%d/%s", port, "getNodeStats")

	response, err := http.Get(url)

	if err != nil {
		fmt.Println(err.Error())
		return discovery.NodeStats{}
	}

	fogNodes := discovery.NodeStats{}
	err = json.NewDecoder(response.Body).Decode(&fogNodes)
	if err != nil {
		fmt.Println(err.Error())
		return discovery.NodeStats{}
	}
	response.Body.Close()
	return fogNodes
}*/
