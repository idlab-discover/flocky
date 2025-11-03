package main

import (
	"context"
	"fmt"

	"oamswirly/common/hwresources"
	mloam "oamswirly/mlservice/oam"
	"oamswirly/monitoringapp/config"
	"oamswirly/monitoringapp/wsclient"
	"os"
	"time"

	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)
	ctx := context.Background()

	for {
		select {
		case <-ctx.Done():
			//log.Info("Context canceled, stopping monitoring app")
			return
		default:
			fmt.Print("\033[H\033[2J")
			fmt.Println("Locally known nodes")
			fmt.Println("_______________________________________")
			basicNodes, _ := wsclient.GetLocallyKnownSvcNodes(config.Cfg.DiscoAPIPort)
			for _, node := range basicNodes {
				fmt.Printf("%s at %s\n", node.Name, node.IP)
			}
			fmt.Println("_______________________________________")
			fmt.Println("")

			fmt.Println("Discovered components")
			fmt.Println("_______________________________________")
			components, _ := wsclient.GetKnownComponentDefs(config.Cfg.RepoAPIPort)
			for concreteCmp, components := range components {
				fmt.Printf("  * Abstract component %s\n", concreteCmp)
				for _, component := range components {
					fmt.Printf("   - %s\n", component)
				}
			}
			fmt.Println("_______________________________________")
			fmt.Println("")

			fmt.Println("Extended node summaries")
			fmt.Println("_______________________________________")
			nodes, _ := wsclient.GetKnownNodeSummaries(config.Cfg.RepoAPIPort)
			for _, node := range nodes {
				traitText := ""
				for _, trait := range node.NodeCaps.Caps.SupportedTraits {
					traitText += trait.Metadata.Name + " "
				}
				fmt.Printf("%s at supporting %s traits and %d workload types\n", node.Name, traitText, len(node.NodeCaps.Caps.SupportedWorkloads))

				cpuStats, cpuOK := node.NodeCaps.Caps.Resources.HwStats[hwresources.ResourceCPU].(hwresources.CPUStats)
				memStats, memOK := node.NodeCaps.Caps.Resources.HwStats[hwresources.ResourceMemory].(hwresources.MemoryStats)
				fsStats, fsOK := node.NodeCaps.Caps.Resources.HwStats[hwresources.ResourceFs].(stats.FsStats)
				gpuStats, gpuOK := node.NodeCaps.Caps.Resources.HwStats[mloam.ResourceAccelerator].(mloam.AcceleratorStats)
				if cpuOK {
					fmt.Printf("    CPU %d/%d\n", *cpuStats.UsageNanoCores, *cpuStats.CPUNanoCores)
				}
				if memOK {
					fmt.Printf("    Mem %d/%d\n", *memStats.UsageBytes, *memStats.TotalBytes)
				}
				if fsOK {
					fmt.Printf("    Fs %d/%d\n", *fsStats.UsedBytes, *fsStats.CapacityBytes)
				}
				if gpuOK {
					fmt.Printf("    GPU #%d: %d/%d\n", len(gpuStats.Accelerators), gpuStats.UsedMemoryBytes, gpuStats.TotalMemoryBytes)
				}
				appsStr := ""
				for _, app := range node.NodeApps.Applications {
					appsStr += app.Metadata.Name + " "
				}
				fmt.Printf("    Running %d apps: %s \n", len(node.NodeApps.Applications), appsStr)
			}
			fmt.Println("_______________________________________")
			fmt.Println("")

			time.Sleep(time.Second * 3)
		}
	}
}
