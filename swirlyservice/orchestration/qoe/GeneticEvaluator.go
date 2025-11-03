package qoe

import "oamswirly/common/oam"

type GeneticEvaluator struct {
}

func (eval GeneticEvaluator) IsAcceptableNode(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) bool {
	return true
}

func (eval GeneticEvaluator) CalculateDistance(node oam.NodeSummary, toDeploy oam.Component, hasComponent bool) float32 {
	return 0
}
