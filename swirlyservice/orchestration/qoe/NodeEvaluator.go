package qoe

import (
	"oamswirly/common/hwresources"
	"oamswirly/common/oam"
	"sort"

	log "github.com/sirupsen/logrus"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

var Evaluator NodeEvaluator

type NodeEvaluator interface {
	//OrderByBestComponentProvider(component oam.Component, concreteDef oam.ComponentDef, knownNodes []oam.NodeSummary, alreadyAllocated []oam.Component) []ActiveProvider
	IsAcceptableNode(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) bool
	CalculateDistance(node oam.NodeSummary, toDeploy oam.Component, hasComponent bool) float32
	//RevalidateNode(provider ActiveProvider, component oam.Component, cDef oam.ComponentDef) bool
}

type ActiveProvider struct {
	Distance         float32
	ExistingInstance *oam.SubApplication
	Node             oam.NodeSummary
}

func OrderByBestComponentProvider(component oam.Component, concreteDef oam.ComponentDef, knownNodes []oam.NodeSummary, alreadyAllocated []oam.Component) []ActiveProvider {
	//log.Infof("Finding component provider for %s", component.Name)
	//First, find required traits
	reqTraits := []oam.Trait{}
	for _, trait := range component.Traits {
		if oam.IsRequiredTrait(trait) {
			reqTraits = append(reqTraits, trait)
		}
	}

	//Filter nodes for required traits
	eligibleNodes := []oam.NodeSummary{}
	for _, node := range knownNodes {
		if Evaluator.IsAcceptableNode(node, concreteDef, reqTraits) {
			eligibleNodes = append(eligibleNodes, node)
		}
	}
	if len(eligibleNodes) == 0 {
		return []ActiveProvider{}
	}

	//Order by best ones with component already deployed (based on QoE/traits)
	sortedProviders := []ActiveProvider{}
	for _, node := range eligibleNodes {
		//log.Infof("Trying node %s latency %f", node.Name, node.NetInfo.Latency)
		//Took this out because Distance isn't calculated at this point, needs to be checked later if and only if a Distance trait is present: && node.Distance < config.Cfg.MaxPing
		//except == nil || (node.PrimaryAddress.String() != except.PrimaryAddress.String()) &&
		if existing := hasSuitableComponentAvailable(node, concreteDef, reqTraits); existing != nil {
			//log.Infof("Initial node conditions ok for node %s, already has component %s with required traits", node.Name, component.Type)

			distance := Evaluator.CalculateDistance(node, component, true)
			provider := ActiveProvider{
				Distance:         distance,
				Node:             node,
				ExistingInstance: existing,
			}
			distanceValid := checkDistanceTrait(provider, component.Traits)
			if distanceValid {
				sortedProviders = append(sortedProviders, provider)
			}
		}
	}

	if len(sortedProviders) > 0 {
		return SortNodes(sortedProviders)
	}

	//Failing that, order by best QoE based on traits
	//log.Infof("No existing service found, retrying with closest")
	for _, node := range eligibleNodes {
		//if except == nil || (node.PrimaryAddress.String() != except.PrimaryAddress.String()) {
		distance := Evaluator.CalculateDistance(node, component, false)
		provider := ActiveProvider{
			Distance: distance,
			Node:     node,
		}
		distanceValid := checkDistanceTrait(provider, component.Traits)
		if distanceValid {
			sortedProviders = append(sortedProviders, provider)
		}
	}
	return SortNodes(sortedProviders)
}

func checkResources(concreteComponentInfo oam.ComponentDef, node oam.NodeSummary) bool {
	reqResources := concreteComponentInfo.Spec.Schematic.Definition.Resources.Requests
	nodeResources := node.NodeCaps.Caps.Resources

	cpuStats, _ := nodeResources.HwStats[hwresources.ResourceCPU].(hwresources.CPUStats)
	cpuFree := float64(*cpuStats.CPUNanoCores-*cpuStats.UsageNanoCores) * cpuStats.CPUEquivalent
	if reqResources.Cpu() != nil {
		cpuReq := reqResources.Cpu().AsApproximateFloat64() * 1000000000
		//log.Infof("IsAcceptableNode CPU check %f > %f ?", cpuFree, cpuReq)
		if cpuFree < cpuReq {
			return false
		}
	}

	if reqResources.Memory() != nil {
		//log.Info("IsAcceptableNode Mem check")
		memStats, _ := nodeResources.HwStats[hwresources.ResourceMemory].(hwresources.MemoryStats)
		memFree := *memStats.AvailableBytes
		memReq, ok := reqResources.Memory().AsInt64()
		if !ok {
			log.Error("IsAcceptableNode Mem check failed")
		} else {
			//log.Infof("IsAcceptableNode Mem check %d > %d ?", memFree, memReq)
			if memFree < uint64(memReq) {
				return false
			}
		}
	}

	if reqResources.Storage() != nil {
		//log.Info("IsAcceptableNode Storage check")
		fsStats := nodeResources.HwStats[hwresources.ResourceFs].(stats.FsStats)
		storSize := *fsStats.AvailableBytes
		storReq, ok := reqResources.Storage().AsInt64()
		if !ok {
			log.Error("IsAcceptableNode Storage check failed")
		} else {
			//log.Infof("IsAcceptableNode Storage check %d > %d ?", storSize, storReq)
			if storSize < uint64(storReq) {
				return false
			}
		}
	}

	netBw, set := reqResources[oam.ResourceNetwork]
	if set {
		//log.Info("IsAcceptableNode Net check")
		nwStats := nodeResources.HwStats[hwresources.ResourceFs].(hwresources.NetworkStats)
		netSize := *nwStats.InterfaceBandwidth
		netReq, ok := netBw.AsInt64()
		if !ok {
			log.Error("IsAcceptableNode Net check failed")
		} else {
			//log.Infof("IsAcceptableNode Net check %d > %d ?", netSize, netReq)
			if netSize < uint64(netReq) {
				return false
			}
		}
	}
	return true
}

func hasSuitableComponentAvailable(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) *oam.SubApplication {
	for _, app := range node.NodeApps.Applications {
		for _, component := range app.Spec.Components {
			//This looks iffy, but hear me out: component.Type always has to refer to a "generic" component type, i.e. the BaseComponent, because we can have multiple concrete ones
			//To keep track of which *actual* componentDef a component is running, we use MetaConcreteComponentNameProperty
			if component.Type == concreteComponentInfo.Spec.Schematic.BaseComponent && component.Properties[oam.MetaConcreteComponentName] == concreteComponentInfo.Metadata.Name {
				return &app
			}
		}
	}
	return nil
}

func RevalidateNode(provider ActiveProvider, component oam.Component, cDef oam.ComponentDef) bool {
	//log.Infof("Revalidation potentially changed node %s for %s", provider.Node.Name, component.Name)
	//First, find required traits
	reqTraits := []oam.Trait{}
	for _, trait := range component.Traits {
		if oam.IsRequiredTrait(trait) {
			reqTraits = append(reqTraits, trait)
		}
	}
	distance := Evaluator.CalculateDistance(provider.Node, component, true)
	provider.Distance = distance

	distanceValid := checkDistanceTrait(provider, component.Traits)
	traitsValid := Evaluator.IsAcceptableNode(provider.Node, cDef, reqTraits)
	return distanceValid && traitsValid
}

func checkDistanceTrait(provider ActiveProvider, traits []oam.Trait) bool {
	for _, trait := range traits {
		if trait.Type == string(oam.SoftDistanceLimitTraitType) {
			traitImpl := oam.TraitFromComponentInfo(trait)
			maxDist := traitImpl.(oam.SoftDistanceLimitTrait).Max
			return provider.Distance < float32(maxDist)
		}
	}
	return true
}

func SortNodes(providers []ActiveProvider) []ActiveProvider {
	nodePings := []ActiveProvider{}
	nodePings = append(nodePings, providers...)

	ping := func(p1, p2 *ActiveProvider) bool {
		return p1.Distance < p2.Distance
	}
	By(ping).Sort(nodePings)

	return nodePings
}

type By func(p1, p2 *ActiveProvider) bool

// Sort is a method on the function type, By, that sorts the argument slice according to the function.
func (by By) Sort(pings []ActiveProvider) {
	ps := &pingSorter{
		pings: pings,
		by:    by, // The Sort method's receiver is the function (closure) that defines the sort order.
	}
	sort.Sort(ps)
}

type pingSorter struct {
	pings []ActiveProvider
	by    func(p1, p2 *ActiveProvider) bool // Closure used in the Less method.
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
