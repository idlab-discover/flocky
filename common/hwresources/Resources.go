package hwresources

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/mackerelio/go-osstat/cpu"
	"github.com/mackerelio/go-osstat/memory"
	"github.com/mackerelio/go-osstat/network"
	log "github.com/sirupsen/logrus"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

type ResourceType string

var ResourceCPU ResourceType = "CPU"
var ResourceMemory ResourceType = "Memory"
var ResourceNetwork ResourceType = "Network"
var ResourceFs ResourceType = "FS"

type NodeResources struct {
	//Hardware     *v1.ResourceList    `json:"hardware"`
	HwStats map[ResourceType]interface{} `json:"hwStats"`
	/*CPUStats     *CPUStats      `json:"cpuStats"`
	MemoryStats  *MemoryStats   `json:"memoryStats"`
	NetworkStats *NetworkStats  `json:"networkStats"`
	FsStats      *stats.FsStats `json:"fsStats"`
	EnergyStats  *EnergyStats   `json:"energyStats"`*/
}

type CPUStats struct {
	stats.CPUStats
	CPUEquivalent float64
	CPUNanoCores  *uint64
}

type NetworkStats struct {
	stats.NetworkStats
	InterfaceBandwidth *uint64
}

type MemoryStats struct {
	stats.MemoryStats
	TotalBytes *uint64
}

/*type EnergyStats struct {
	GreenEnergyAvailable *uint64
	GreenEnergyCapacity  *uint64
	CO2kWhEquivalent     *uint64
}*/

/*func GetTotalMemory() int {
	//Get memory
	memFree, _ := ExecCmdBash("free -m | grep 'Mem:'")

	memSize := strings.Split(reInsideWhtsp.ReplaceAllString(memFree, " "), " ")[1]
	memInt, _ := strconv.Atoi(memSize)
	return memInt * 1024 * 1024
}*/

/*func GetTotalStorage() (uint64, error) {
	//Get disk
	diskFree, err := ExecCmdBash("df  | grep -E '[[:space:]]/$'")
	if err != nil {
		return 0, err
	}

	diskSize := strings.Split(reInsideWhtsp.ReplaceAllString(diskFree, " "), " ")[1]
	diskBytes, err := strconv.ParseUint(diskSize, 10, 64)
	if err != nil {
		return 0, err
	}
	return diskBytes, nil
}*/

func GetUsedStorage() (uint64, error) {
	//Get disk
	diskFree, err := ExecCmdBash("df -B 1 | grep -E '[[:space:]]/$'")
	if err != nil {
		return 0, err
	}

	diskSize := strings.Split(reInsideWhtsp.ReplaceAllString(diskFree, " "), " ")[2]
	diskBytes, err := strconv.ParseUint(diskSize, 10, 64)
	if err != nil {
		return 0, err
	}
	return diskBytes, nil
}

func GetFreeStorage() (uint64, error) {
	//Get disk
	diskFree, err := ExecCmdBash("df -B 1 | grep -E '[[:space:]]/$'")
	if err != nil {
		return 0, err
	}

	diskSize := strings.Split(reInsideWhtsp.ReplaceAllString(diskFree, " "), " ")[3]
	diskBytes, err := strconv.ParseUint(diskSize, 10, 64)
	if err != nil {
		return 0, err
	}
	return diskBytes, nil
}

/*func GetCores() int {
	stdout, _ := ExecCmdBash("nproc")
	numCpus := strings.Trim(string(stdout), "\n")

	cpusInt, _ := strconv.Atoi(numCpus)
	return cpusInt
}*/

func ExecCmdBash(dfCmd string) (string, error) {
	//log.Infof("Executing %s", dfCmd)
	cmd := exec.Command("sh", "-c", dfCmd)
	stdout, err := cmd.Output()

	if err != nil {
		println(err.Error())
		return "", err
	}
	return string(stdout), nil
}

/*func GetFreeMemory() int {
	//Get memory
	//there's different types of output of the free command, trying the one with -/+ buffers/cache: first
	memFree, _ := ExecCmdBash("free -m | grep 'cache:'")
	var memSize string
	if memFree != "" {
		memSize = strings.Split(reInsideWhtsp.ReplaceAllString(memFree, " "), " ")[2]
	} else {
		memFree, _ = ExecCmdBash("free -m | grep 'Mem:'")
		memSize = strings.Split(reInsideWhtsp.ReplaceAllString(memFree, " "), " ")[2]
	}
	memMb, _ := strconv.Atoi(memSize)
	return int(memMb) * 1024 * 1024
}*/

/*func GetCPU() int {
	iostatc, _ := ExecCmdBash("iostat -c")
	var cpuUsed string
	var cpuLines = strings.Split(iostatc, "\n")

	cpuUsed = strings.Split(reInsideWhtsp.ReplaceAllString(cpuLines[3], " "), " ")[6]

	cpuPct, _ := strconv.ParseFloat(cpuUsed, 64)
	return int((100 - cpuPct) * float64(GetCores()) / 100)
}*/

var reInsideWhtsp = regexp.MustCompile(`\s+`)

var totalNanoCores uint64
var userCorePct, systemCorePct, totalCorePct uint64
var numCores int

func CPUStatLoop() {
	for {
		before, err := cpu.Get()
		if err != nil {
			log.Errorf("Couldn't get CPU stats %v", err)
			return
		}
		time.Sleep(time.Duration(1) * time.Second)
		after, err := cpu.Get()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err)
			return
		}
		//total := float64(after.Total - before.Total)
		userCorePct = after.User - before.User
		systemCorePct = after.System - before.System
		numCores = after.CPUCount
		//fmt.Printf("cpu idle: %f %%\n", float64(after.Idle-before.Idle)/total*100)
	}
}

func GetSimplifiedResources() (map[v1.ResourceName]int, map[v1.ResourceName]int) {
	resources := make(map[v1.ResourceName]int)
	totalResources := make(map[v1.ResourceName]int)

	memory, err := memory.Get()
	if err != nil {
		log.Errorf("Couldn't get memory stats %v", err)
		return resources, totalResources
	}

	/*cpu, err := cpu.Get()
	if err != nil {
		log.Errorf("Couldn't get CPU stats %v", err)
		return resources, totalResources
	}*/

	resources[v1.ResourceCPU] = int(systemCorePct + userCorePct)
	resources[v1.ResourceMemory] = int(memory.Used) //GetFreeMemory()

	totalResources[v1.ResourceCPU] = numCores * 100 //GetCores()
	totalResources[v1.ResourceMemory] = int(memory.Total)
	return resources, totalResources
}

func GetResources() NodeResources {
	cpuStats, _ := GetCPUStats()
	//bailout for older free versions, in which case this is more accurate for "available" memory
	memStats, _ := GetMemStats()
	netStats, _ := GetNetStats()
	fsStats, _ := GetFsStats()
	//enStats, _ := GetEnergyStats()
	//TODO from here on for fs stats

	/*hwspecs := v1.ResourceList{}
	hwspecs[v1.ResourceCPU], _ = resource.ParseQuantity(fmt.Sprintf("%d", GetCores()))
	hwspecs[v1.ResourceMemory], _ = resource.ParseQuantity(fmt.Sprintf("%d", GetTotalMemory()))
	hwspecs[v1.ResourceStorage], _ = resource.ParseQuantity(fmt.Sprintf(GetTotalStorage()))
	hwspecs[oam.ResourceCPUEquivalent], _ = resource.ParseQuantity("1")
	hwspecs[oam.ResourceNetwork], _ = resource.ParseQuantity("1000")*/

	stats := make(map[ResourceType]interface{})
	stats[ResourceCPU] = cpuStats
	stats[ResourceMemory] = memStats
	stats[ResourceNetwork] = netStats
	stats[ResourceFs] = fsStats

	resources := NodeResources{
		//Hardware:     &hwspecs,
		HwStats: stats,
	}

	return resources
}

/*func GetEnergyStats() (EnergyStats, error) {
	return EnergyStats{}, nil
}*/

func GetFsStats() (stats.FsStats, error) {
	freeDisk, _ := GetFreeStorage()
	//totalDisk, _ := GetTotalStorage()
	usedDisk, _ := GetUsedStorage()
	totalDisk := usedDisk + freeDisk
	//log.Infof("Fs stats free %d used %d total %d", freeDisk, usedDisk, totalDisk)
	return stats.FsStats{
		AvailableBytes: &freeDisk,
		CapacityBytes:  &totalDisk,
		UsedBytes:      &usedDisk,
	}, nil
}

func GetNetStats() (NetworkStats, error) {
	/*ifacesStr, err := ExecCmdBash("ip a | grep -o -E '[0-9]{1,2}: [a-z0-9]*: ' | grep -o -E '[a-z0-9]{2,}'") //cmd.Output()
	if err != nil {
		log.Errorf("getResources Net error %v", err)
		return NetworkStats{}, err
	}

	ifaces := strings.Split(ifacesStr, "\n")

	ifacesStats := []stats.InterfaceStats{}
	for _, iface := range ifaces {
		ifacesStatsStr, err := ExecCmdBash("ifconfig " + iface + "| grep 'bytes'") //cmd.Output()
		if err != nil {
			log.Errorf("getResources Net error %v", err)
			return NetworkStats{}, err
		}

		//log.Info(ifacesStatsStr)
	}*/
	ifacesStats := []stats.InterfaceStats{}
	ifsStats, err := network.Get()
	if err != nil {
		log.Errorf("Couldn't get net stats %v", err)
	}
	for _, ifStats := range ifsStats {
		ifacesStats = append(ifacesStats, stats.InterfaceStats{
			Name:    ifStats.Name,
			RxBytes: &ifStats.RxBytes,
			TxBytes: &ifStats.TxBytes,
		})
	}

	//TODO
	var ifBw uint64 = 1000000000
	netStats := NetworkStats{
		NetworkStats: stats.NetworkStats{
			Time:       metav1.Now(),
			Interfaces: ifacesStats,
		},
		InterfaceBandwidth: &ifBw,
	}
	return netStats, nil
}

func GetMemStats() (MemoryStats, error) {
	//cmd := exec.Command("free | grep 'Mem:'")
	/*memStatsBytes, err := ExecCmdBash("free -b | grep 'Mem:'") //cmd.Output()
	memUsed := uint64(0)
	memFree := uint64(0)
	memSize := uint64(0)

	if err != nil {
		log.Errorf("getResources Mem error %v", err)
		return MemoryStats{}, err
	}
	memStatsStr := string(memStatsBytes)

	cats := strings.Split(reInsideWhtsp.ReplaceAllString(memStatsStr, " "), " ")
	memFree, err = strconv.ParseUint(cats[6], 10, 64)
	if err != nil {
		log.Errorf("getResources Mem error %v", err)
		return MemoryStats{}, err
	}
	memSize, err = strconv.ParseUint(cats[1], 10, 64)
	if err != nil {
		log.Errorf("getResources Mem error %v", err)
		return MemoryStats{}, err
	}

	memStatsStr, err = ExecCmdBash("free -b | grep '+'")
	if err != nil {
		log.Errorf("getResources Mem error %v", err)
		//return stats.MemoryStats{}, err
	}

	if memStatsStr != "" {
		cats := strings.Split(reInsideWhtsp.ReplaceAllString(memStatsStr, " "), " ")
		memFree, err = strconv.ParseUint(cats[2], 10, 64)
		if err != nil {
			log.Errorf("getResources Mem error %v", err)
			return MemoryStats{}, err
		}
	}

	memUsed = memSize - memFree*/
	memory, err := memory.Get()
	if err != nil {
		log.Errorf("Couldn't get memory stats %v", err)
		return MemoryStats{}, err
	}

	//log.Infof("Mem stats used %d available %d size %d", memory.Used, memory.Free, memory.Total)
	memStats := MemoryStats{
		MemoryStats: stats.MemoryStats{
			Time:            metav1.Now(),
			UsageBytes:      &memory.Used,
			AvailableBytes:  &memory.Free,
			WorkingSetBytes: &memory.Total,
		},
		TotalBytes: &memory.Total,
	}
	return memStats, nil
}

func GetCPUStats() (CPUStats, error) {
	/*cpuStatsStr, err := ExecCmdBash("mpstat 1 1 | grep 'all'") //cmd.Output()
	if err != nil {
		log.Errorf("getResources CPU error %v", err)
		return CPUStats{}, err
	}

	nProc, err := ExecCmdBash("nproc")
	if err != nil {
		log.Errorf("getResources CPU error %v", err)
		return CPUStats{}, err
	}

	numCpus, err := strconv.Atoi(strings.Trim(nProc, "\n"))
	if err != nil {
		log.Errorf("getResources CPU error %v", err)
		return CPUStats{}, err
	}

	cpuStatsLines := strings.Split(cpuStatsStr, "\n")

	cpuCats := strings.Split(reInsideWhtsp.ReplaceAllString(cpuStatsLines[0], " "), " ")
	cpuIdle, err := strconv.ParseFloat(cpuCats[len(cpuCats)-1], 64)
	if err != nil {
		log.Errorf("getResources CPU error %v", err)
		return CPUStats{}, err
	}*/

	cpuNanos := (userCorePct + systemCorePct) * 10000000

	totalNanoCores += cpuNanos
	installedNanoCores := uint64(numCores * 1000000000)

	//log.Infof("CPU stats cores %d nanos installed %d nanos used %d", numCores, installedNanoCores, cpuNanos)
	cpuStats := CPUStats{
		CPUStats: stats.CPUStats{
			Time:                 metav1.Now(),
			UsageNanoCores:       &cpuNanos,
			UsageCoreNanoSeconds: &totalNanoCores,
		},
		CPUNanoCores: &installedNanoCores,
	}
	return cpuStats, nil
}
