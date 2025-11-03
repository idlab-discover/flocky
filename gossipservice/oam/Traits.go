package oam

import "oamswirly/common/oam"

var GossipTraitType oam.TraitType = "discover.flocky.oam.Gossip"
var SvcName oam.FlockyService = "discover.flocky.oam.GossipSvc"

var ReadRESTGossipProperty string = "ReadRESTGossip"
var ReadShmGossipProperty string = "ReadShmGossip"
var WriteRESTGossipProperty string = "WriteRESTGossip"
var WriteShmGossipProperty string = "WriteShmGossip"

func GetGossipTrait() oam.TraitDef {
	return oam.CreateTraitDef(string(GossipTraitType), oam.GetSwirlySupportedWorkloadTypesAlt())
}

func GetReadRESTGossipSettings(trait oam.Trait) []RESTSettings {
	if settings, exists := trait.Properties[ReadRESTGossipProperty]; exists {
		return settings.([]RESTSettings)
	}
	return []RESTSettings{}
}

func GetWriteRESTGossipSettings(trait oam.Trait) []RESTSettings {
	if settings, exists := trait.Properties[WriteRESTGossipProperty]; exists {
		return settings.([]RESTSettings)
	}
	return []RESTSettings{}
}

func GetReadShmGossipSettings(trait oam.Trait) []ShmSettings {
	if settings, exists := trait.Properties[ReadShmGossipProperty]; exists {
		return settings.([]ShmSettings)
	}
	return []ShmSettings{}
}

func GetWriteShmGossipSettings(trait oam.Trait) []ShmSettings {
	if settings, exists := trait.Properties[WriteShmGossipProperty]; exists {
		return settings.([]ShmSettings)
	}
	return []ShmSettings{}
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
