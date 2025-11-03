package clients

import "oamswirly/common/oam"

var Orch Orchestrator

type Orchestrator interface {
	Init() Orchestrator
	DeploySubApp(pod oam.SubApplication) bool
	RemoveSubApp(pod oam.SubApplication) bool
}
