package qoe

import (
	"oamswirly/common/oam"
)

type LegacyEvaluator struct {
}

func (eval LegacyEvaluator) IsAcceptableNode(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) bool {
	//Check available resources first
	//log.Infof("IsAcceptableNode check %s", node.Name)
	if !checkResources(concreteComponentInfo, node) {
		return false
	}

	//valid := true
	for _, trait := range requiredTraits {
		//traitImpl := oam.TraitFromComponentInfo(trait)
		if !oam.CheckNodeTrait(node, trait) {
			//log.Infof("IsAcceptableNode required trait %s not present", trait.Type)
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
	//log.Infof("IsAcceptableNode required workload type %s present? %t", wlType, wlCheck)
	return wlCheck
}

func (eval LegacyEvaluator) CalculateDistance(node oam.NodeSummary, toDeploy oam.Component, hasComponent bool) float32 {
	return node.NetInfo.Latency
}
