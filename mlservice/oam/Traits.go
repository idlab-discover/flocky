package oam

import "oamswirly/common/oam"

var MLInferenceTraitType oam.TraitType = "discover.flocky.oam.MLInference"
var MLAcceleratorTraitType oam.TraitType = "discover.flocky.oam.MLAccelerator"
var MLConfidentialComputingTraitType oam.TraitType = "discover.flocky.oam.MLConfidentialComputing"
var MLGossipLearningTraitType oam.TraitType = "discover.flocky.oam.MLGossipLearning"

var ReadRESTProperty string = "ReadREST"
var ReadShmProperty string = "ReadShm"
var WriteRESTProperty string = "WriteREST"
var WriteShmProperty string = "WriteShm"
var LearningType string = "LearningType"

type LearningAlgo string

var GossipLearning LearningAlgo = "Gossip"
var DeltaSumLearning LearningAlgo = "DeltaSum"
var VarianceCorrLearning LearningAlgo = "VarianceCorrect"

func GetMachineLearningTrait(specificTrait oam.TraitType) oam.TraitDef {
	return oam.CreateTraitDef(string(specificTrait), []oam.WorkloadDef{oam.GetContainerServiceType()})
}

func GetNodeSupportedTraits() []oam.TraitDef {
	traits := []oam.TraitDef{
		GetMachineLearningTrait(MLInferenceTraitType),
		GetMachineLearningTrait(MLGossipLearningTraitType),
	}
	if HasAccelerators() {
		traits = append(traits, GetMachineLearningTrait(MLAcceleratorTraitType))
	}
	if HasConfComputing() > 0 {
		traits = append(traits, GetMachineLearningTrait(MLConfidentialComputingTraitType))
	}

	return traits
}

func GetReadRESTSettings(trait oam.Trait) []RESTSettings {
	if settings, exists := trait.Properties[ReadRESTProperty]; exists {
		return settings.([]RESTSettings)
	}
	return []RESTSettings{}
}

func GetWriteRESTSettings(trait oam.Trait) []RESTSettings {
	if settings, exists := trait.Properties[WriteRESTProperty]; exists {
		return settings.([]RESTSettings)
	}
	return []RESTSettings{}
}

func GetReadShmSettings(trait oam.Trait) []ShmSettings {
	if settings, exists := trait.Properties[ReadShmProperty]; exists {
		return settings.([]ShmSettings)
	}
	return []ShmSettings{}
}

func GetWriteShmSettings(trait oam.Trait) []ShmSettings {
	if settings, exists := trait.Properties[WriteShmProperty]; exists {
		return settings.([]ShmSettings)
	}
	return []ShmSettings{}
}

func GetLearningType(trait oam.Trait) LearningAlgo {
	if settings, exists := trait.Properties[WriteShmProperty]; exists {
		return settings.(LearningAlgo)
	}
	return ""
}

type RESTSettings struct {
	Key      string
	Endpoint string
}

type ShmSettings struct {
	Key         string
	Size        int
	ChannelName string `json:"channelName"`
	ChannelPath string `json:"channelPath"`
	SemaWait    string `json:"semaWait"`
	SemaSignal  string `json:"semaSignal"`
}
