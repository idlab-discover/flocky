package main

import (
	"encoding/json"
	"fmt"
	_ "image/png"
	"math"
	"net/http"
	discovery "oamswirly/common/nodediscovery"
	"oamswirly/runner_deltasum/config"
	"os"
	"os/exec"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"syscall"
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
		//fnLimit := float64(RoundUp(float64(en) / 1000))
		for fn := config.Cfg.MinFogNodes; fn <= config.Cfg.MaxFogNodes; fn += config.Cfg.FogNodeStep {
			outfilename_python := fmt.Sprintf("f%d-e%dmetrics.csv", fn, en)
			fmt.Printf("Renewing metrics file output to %s\n", outfilename_python)
			out_python, _ := os.Create(outfilename_python)
			fileFDs = append(fileFDs, out_python)

			outfilename_gossip := fmt.Sprintf("gossip_metrics_f%d-e%d.csv", fn, en)
			fmt.Printf("Renewing metrics file output to %s\n", outfilename_gossip)
			out_gossip, _ := os.Create(outfilename_gossip)
			fileFDs = append(fileFDs, out_gossip)

			for iter := 0; iter < config.Cfg.Iterations; iter++ {
				folderPath := fmt.Sprintf("f%d-e%d", fn, en)
				fmt.Printf("Fog %d edge %d iteration %d, using files in %s", fn, en, iter, folderPath)

				os.MkdirAll(folderPath, os.ModePerm)

				fogPids, edgePids := startNodes(fn, en, folderPath)

				sigc := make(chan os.Signal, 1)
				signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
				go func() {
					<-sigc
					//log.Infof("Signal %v detected, shutting down", sig.String())
					stopNodes(fogPids, edgePids)
					os.Exit(0)
				}()

				monitorNodes()
				time.Sleep(time.Duration(fn) * 30 * time.Minute)
				// time.Sleep(30 * time.Second)
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
	fogPids := make(map[string]int)
	edgePids := make(map[string]int)

	if numFogs > 1 {
		for nr := 0; nr < numFogs; nr++ {
			cfgFile := fmt.Sprintf("%s/fog%d.json", prefix, nr)
			gossipCfgFile := fmt.Sprintf("%s/gossip%d.json", prefix, nr)
			fmt.Printf("Starting fog node from %s\n", cfgFile)

			pids, err := startNode(cfgFile, gossipCfgFile, prefix)
			if err != nil {
				fmt.Printf("Failed to start fog node %d: %s\n", nr, err.Error())
			} else {
				for name, pid := range pids {
					fogPids[fmt.Sprintf("f_%s_%d", name, nr)] = pid
				}
				fmt.Printf("Started processes for fog node %d: %v\n", nr, pids)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}

	/*if numEdges > 1 {
		for nr := 0; nr < numEdges; nr++ {
			cfgFile := fmt.Sprintf("%s/edge%d.json", prefix, nr)
			fmt.Printf("Starting edge node from %s\n", cfgFile)

			pids, err := startNode(cfgFile, prefix)
			if err != nil {
				fmt.Printf("Failed to start edge node %d: %s\n", nr, err.Error())
			} else {
				for name, pid := range pids {
					edgePids[fmt.Sprintf("f_%s_%d", name, nr)] = pid
				}
			}
			time.Sleep(200 * time.Millisecond)
		}
	}*/

	return fogPids, edgePids
}

func monitorNodes() { //fogPids map[string]int, edgePids map[string]int) {
	// fogData := make(map[string][]StatsLine)
	// edgeData := make(map[string][]StatsLine)
	// generalData := []StatsLine{}

	// for node, _ := range fogPids {
	// 	fogData[node] = []StatsLine{}
	// }

	// for node, _ := range edgePids {
	// 	edgeData[node] = []StatsLine{}
	// }

	// fmetricNames := []string{"memory", "cpu"} //, "neighbours", "accuracy", "inrange", "tolerated", "outsiderange"}
	// emetricNames := []string{"memory", "cpu", "minping", "ping"}
	// prevCpu := make(map[string]int)
	prevNetTraffic := 0
	prevLoNetTraffic := 0
	timeTaken := int64(0)
	for iteration := 0; iteration < config.Cfg.MonitorLoops; iteration++ {
		//fmt.Printf("Monitor iteration %d, previous time taken %d\n", iteration, timeTaken)
		time.Sleep(time.Duration(config.Cfg.MonitorPeriod*1000-timeTaken) * time.Millisecond)
		startTime := time.Now()

		// totalDiscovered := 0
		// totalExpected := 0

		// for node, pid := range fogPids {
		// 	metrics := make(map[string]float64)
		// 	metrics["memory"] = float64(getMemory(pid))

		// 	prevNCpu, found := prevCpu[node]
		// 	cpuvalue := float64(getCPU(pid))
		// 	if !found {
		// 		metrics["cpu"] = 0
		// 	} else {
		// 		metrics["cpu"] = (cpuvalue - float64(prevNCpu)) / float64(config.Cfg.MonitorPeriod)
		// 	}
		// 	prevCpu[node] = int(cpuvalue)

		// 	//neighbours := getKnownFogNodes(node)
		// 	//stats := getDiscoveredNodeStats(node)
		// 	//neighbours := float64(stats.NodesWithinRange + stats.NodesInAcceptableRange + stats.OutsideRange)
		// 	metrics["neighbours"] = 1 //neighbours

		// 	//acc := math.Min(float64(stats.Discovered)/math.Max(1, float64(stats.ExpectedInRange)), 1)

		// 	metrics["accuracy"] = 1     // acc
		// 	metrics["inrange"] = 1      //float64(stats.NodesWithinRange) / neighbours
		// 	metrics["tolerated"] = 0    //float64(stats.NodesInAcceptableRange) / neighbours
		// 	metrics["outsiderange"] = 0 //float64(stats.OutsideRange) / neighbours

		// 	totalDiscovered += 1 // stats.Discovered
		// 	totalExpected += 1   //stats.ExpectedInRange

		// 	line := StatsLine{
		// 		MNames:  fmetricNames,
		// 		Metrics: metrics,
		// 	}
		// 	fogData[node] = append(fogData[node], line)

		// }

		// serviced := 0
		// fogservers := make(map[string]bool)
		// for node, pid := range edgePids {
		// 	metrics := make(map[string]float64)
		// 	metrics["memory"] = float64(getMemory(pid))

		// 	prevNCpu, found := prevCpu[node]
		// 	cpuvalue := float64(getCPU(pid))
		// 	if !found {
		// 		metrics["cpu"] = 0
		// 	} else {
		// 		metrics["cpu"] = cpuvalue - float64(prevNCpu)
		// 	}
		// 	prevCpu[node] = int(cpuvalue)

		// 	//stats := getNodeStats(node)

		// 	/*metrics["minping"] = float64(stats.MinimalPing)
		// 	metrics["ping"] = float64(stats.CurrentClosestPing)
		// 	if stats.CurrentClosestPing > 0 {
		// 		serviced++
		// 		fogservers[stats.CurrentFogNode] = true
		// 	}*/

		// 	line := StatsLine{
		// 		MNames:  emetricNames,
		// 		Metrics: metrics,
		// 	}
		// 	edgeData[node] = append(edgeData[node], line)
		// }

		totNetTraffic := float64(0)
		//gMetrics := make(map[string]float64)
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

		// gMetrics["network"] = totNetTraffic
		// gMetrics["cpu"] = float64(getTotalCPU())
		// gMetrics["discovered"] = float64(totalDiscovered)
		// gMetrics["expected"] = float64(totalExpected)
		// gMetrics["serviced"] = float64(serviced)
		// gMetrics["fognodes"] = float64(len(fogservers))

		// generalLine := StatsLine{
		// 	MNames:  []string{"network", "cpu", "discovered", "expected", "serviced", "fognodes"},
		// 	Metrics: gMetrics,
		// }
		// generalData = append(generalData, generalLine)
		fmt.Printf("%d,%f\n", iteration, totNetTraffic)
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

	// if len(fogData) > 0 {
	// 	fogTimeGroups := []GroupStats{}
	// 	for i := 0; i < config.Cfg.MonitorLoops; i++ {
	// 		groupLines := []StatsLine{}
	// 		for _, nodelines := range fogData {
	// 			if nodelines[i].Metrics["accuracy"] > 0 {
	// 				groupLines = append(groupLines, nodelines[i])
	// 			}
	// 			//fmt.Println(data)
	// 		}
	// 		fogTimeGroups = append(fogTimeGroups, MakeGroupStats(groupLines))
	// 	}

	// 	fmt.Println(fogTimeGroups[0].GroupHeader())
	// 	for i := 0; i < len(fogTimeGroups); i++ {
	// 		data, _ := fogTimeGroups[i].String()
	// 		fmt.Println(data)
	// 	}
	// 	fmt.Println("")
	// }

	// if len(edgeData) > 0 {
	// 	edgeTimeGroups := []GroupStats{}
	// 	for i := 0; i < config.Cfg.MonitorLoops; i++ {
	// 		groupLines := []StatsLine{}
	// 		for _, nodelines := range edgeData {
	// 			if nodelines[i].Metrics["ping"] > 0 {
	// 				groupLines = append(groupLines, nodelines[i])
	// 			}
	// 		}
	// 		edgeTimeGroups = append(edgeTimeGroups, MakeGroupStats(groupLines))
	// 	}

	// 	fmt.Println(edgeTimeGroups[0].GroupHeader())
	// 	for i := 0; i < len(edgeTimeGroups); i++ {
	// 		data, _ := edgeTimeGroups[i].String()
	// 		fmt.Println(data)
	// 	}
	// 	fmt.Println("")
	// }

	// fmt.Println(generalData[0].LineHeader())
	// for i := 0; i < len(generalData); i++ {
	// 	data, _ := generalData[i].String()
	// 	fmt.Println(data)
	// }
	// fmt.Println("")
}

func stopNodes(fogPids map[string]int, edgePids map[string]int) {
	for _, pid := range fogPids {
		execCmdBash(fmt.Sprintf("kill %d", pid))
	}

	for _, pid := range edgePids {
		execCmdBash(fmt.Sprintf("kill %d", pid))
	}
}

func startProcess(executable string, cfg string, prefix string, processName string, arg string) (int, error) {
	parts := strings.Split(cfg, "/")
	fname := parts[len(parts)-1]
	outfilename := fmt.Sprintf("%s/out_%s_%s.txt", prefix, fname, processName)
	fmt.Printf("Logging %s output to %s\n", processName, outfilename)
	outfile, _ := os.Create(outfilename)
	fileFDs = append(fileFDs, outfile)

	var cmd *exec.Cmd

	if arg != "" {
		cmd = exec.Command(executable, "-u", arg, cfg, prefix)
	} else {
		cmd = exec.Command(executable, cfg, prefix)
	}
	cmd.Stdout = outfile
	cmd.Stderr = outfile

	err := cmd.Start()
	if err != nil {
		return 0, err
	}
	return cmd.Process.Pid, nil
}

func startNode(discocfg string, gossipcfg string, prefix string) (map[string]int, error) {
	pids := make(map[string]int)

	// Start the discovery service
	discoveryPid, err := startProcess("./discoverysvc", discocfg, prefix, "discoverysvc", "")
	if err != nil {
		return nil, fmt.Errorf("failed to start discovery service: %v", err)
	}
	pids["discoverysvc"] = discoveryPid
	// Start the Python script
	pythonPid, err := startProcess("../.venv/bin/python", discocfg, prefix, "python", "main.py")
	if err != nil {
		return nil, fmt.Errorf("failed to start Python script: %v", err)
	}
	pids["python"] = pythonPid
	// Start your custom Go service
	//if gossipcfg != "f10-e1/gossip9.json" {
	gossipPid, err := startProcess("./gossipsvc", gossipcfg, prefix, "gossipsvc", "")
	if err != nil {
		return nil, fmt.Errorf("failed to start custom Go service: %v", err)
	}
	pids["gossipsvc"] = gossipPid
	//}

	return pids, nil
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
