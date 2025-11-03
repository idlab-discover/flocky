package qoe

import "oamswirly/common/oam"

type LearningEvaluator struct {
}

func (eval LearningEvaluator) IsAcceptableNode(node oam.NodeSummary, concreteComponentInfo oam.ComponentDef, requiredTraits []oam.Trait) bool {
	return true
}

func (eval LearningEvaluator) CalculateDistance(node oam.NodeSummary, toDeploy oam.Component, hasComponent bool) float32 {
	return 0
}
