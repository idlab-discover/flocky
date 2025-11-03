package nodediscovery

import "oamswirly/common/oam"

//"so-swirly/fogservice/config"

/*type NodePinger interface {
	Init() NodePinger
	OrderKnownNodes([]FogNode) []FogNode
	GetNodeType() NodeType
	GetPingURL(ip string, node string) string
	GetFogURL(ip string, node string) string
	GetPingThreshold(nearbyNodes int) float32
	ShouldReping(node FogNode) bool
	GetNodeID() string
}*/

type SvcNode struct {
	Name         string  `json:"name"`
	IP           string  `json:"ip"`
	Distance     float32 `json:"distance"`
	DiscoAPIPort int     `json:"discoAPIPort"`
	RepoAPIPort  int     `json:"repoAPIPort"`
}

type NodesUpdate struct {
	NewNodes     []SvcNode
	DeletedNodes []SvcNode
}

type DiscoveredNodes struct {
	NodesWithinRange       int
	NodesInAcceptableRange int
	ExpectedInRange        int
	OutsideRange           int
	Discovered             int
}

type NodeStats struct {
	CurrentClosestPing float32
	MinimalPing        float32
	CurrentFogNode     string
}

type Client struct {
	Name         string
	Type         NodeType
	RepoAPIPort  int
	DiscoAPIPort int
}

type ServiceClient struct {
	Name string
	//Component         oam.Component
	//ConcreteComponent oam.ComponentDef
	Application oam.SubApplication
}

type UpdateListener struct {
	Name string
	Port string
}

type NodeType string

//var NodeTypeFog NodeType = "fognode"
//var NodeTypeEdge NodeType = "edgenode"

var NodeTypeService NodeType = "servicenode"
var NodeTypeClient NodeType = "clientnode"
