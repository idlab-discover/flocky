package qoe

import (
	"math"
	"oamswirly/common/hwresources"
	"oamswirly/common/oam"

	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

type StrictLegacyEvaluator struct {
}

func (eval StrictLegacyEvaluator) IsAcceptableNode(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) bool {
	//Check available resources first
	//log.Infof("IsAcceptableNode check %s", node.Name)
	if !checkResources(concreteComponentInfo, node) {
		return false
	}

	//valid := true
	for _, trait := range requiredTraits {
		//traitImpl := oam.TraitFromComponentInfo(trait)
		if !oam.CheckNodeTrait(node, trait) {
			return false
		}
	}
	//Traits cover pretty much everything, except explicitly the required workload type..
	wlCheck := false
	wlType := concreteComponentInfo.Spec.Workload.Type
	for _, wlDef := range node.NodeCaps.Caps.SupportedWorkloads {
		if wlDef.Metadata.Name == wlType {
			wlCheck = true
		}
	}
	return wlCheck
}

func (eval StrictLegacyEvaluator) CalculateDistance(node oam.NodeSummary, component oam.Component, hasComponent bool) float64 {
	qualityDistance := float64(0)
	//Missing optional traits have the highest penalty factor, we absolutely want to order by this
	//optTraits := []oam.Trait{}
	for _, trait := range component.Traits {
		if !oam.IsRequiredTrait(trait) {
			//traitImpl := oam.TraitFromComponentInfo(trait)
			if !oam.CheckNodeTrait(node, trait) {
				qualityDistance += 10000
			}
		}
	}
	//Next up is the nextcomponent bool, straight down the middle of missing trait penalty.
	if !hasComponent {
		qualityDistance += 5000
	}

	nodeResources := node.NodeCaps.Caps.Resources
	//After that, we take the resources formula from SoSmarty
	//36S(m, 1024) + 400S(b, 100)
	cpuStats, _ := nodeResources.HwStats[hwresources.ResourceCPU].(hwresources.CPUStats)
	memStats, _ := nodeResources.HwStats[hwresources.ResourceMemory].(hwresources.MemoryStats)
	fsStats := nodeResources.HwStats[hwresources.ResourceFs].(stats.FsStats)
	nwStats := nodeResources.HwStats[hwresources.ResourceFs].(hwresources.NetworkStats)
	cpu := float64(*cpuStats.CPUNanoCores-*cpuStats.UsageNanoCores) * cpuStats.CPUEquivalent
	mem := float64(*memStats.AvailableBytes)
	net := float64(*nwStats.InterfaceBandwidth)
	stor := float64(*fsStats.AvailableBytes)

	//Some explanation required, and these numbers should be extracted to config ASAP
	//Mem: we want to aim for devices with 2GB of free memory, any more gets diminishing returns. Divide mem by 2*1024^3, multiply by 0.0000014 to get max 1000 score with no free mem.
	//Bandwidth: this is an iffy one because IoT stuff shouldn't use much anyway. Aim for 10Mbit connection, 1000 max score.
	//CPU: many of these things have 2 cores now, but one will get a good score too. Remember though, this is unused CPU cores. Max score 1000.
	//Storage: no clue. Let's aim for 4GB free, 500 score. I would say "what idiot will ever deploy a 4GB container on an IoT device", but that might turn out to be me.
	qualityDistance += 0.00000933*weightedSigmoid(mem, 2*1024*1024*1024) + 0.0002*weightedSigmoid(net, 10000000) + 1000*weightedSigmoid(cpu, 2) + 0.0000001164*weightedSigmoid(stor, 4*1024*1024*1024)

	//And finally add latency factor
	qualityDistance += 10 * float64(node.NetInfo.Latency)

	//Funny thing is, the way energystats are provided we don't even need to check the node for green energy trait here
	//Even funnier thing is: the node can have green energy capacity (and the trait), but not have any green energy available right now
	//Let's not dwell on that, most of that stuff is volatile AF anyway
	//Let's pretend batteries don't exist either. Mine might as well not in winter.
	//Max penalty for green energy: 200, or straight up CO2/kWh eq if no green available. The latter may be really high in some areas.
	/*if *nodeResources.EnergyStats.GreenEnergyAvailable > uint64(0) {
		qualityDistance += float64(2000-*nodeResources.EnergyStats.GreenEnergyAvailable) / 10
	} else {
		qualityDistance += float64(*nodeResources.EnergyStats.CO2kWhEquivalent)
	}*/

	//For those counting, this method was crafted so that the combined resource penalties (or rather, lack thereof) can't actually defeat a node
	//where the component is already available, even if available resources are really good.
	//Unless you live on Mars (latency) or in Germany (dirty ass electricity). Cases with extreme latency and dirty electricity should be avoided anyway.
	return qualityDistance

}

func sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func weightedSigmoid(x float64, s float64) float64 {
	return (float64(1) - sigmoid(6*x/s)) * s
}
