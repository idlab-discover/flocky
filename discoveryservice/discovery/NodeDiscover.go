package discovery

import (
	"context"
	"math"
	"net/http"
	"oamswirly/common/nodediscovery"
	"oamswirly/discoveryservice/config"
	"oamswirly/discoveryservice/discovery/wsclient"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

type NodePinger interface {
	//Init() NodePinger
	//GetNodeType() NodeType
	GetPingThreshold(nearbyNodes int) float32
	ShouldReping(node nodediscovery.SvcNode) bool
	//GetNodeID() string
}

var discoveredNodes map[string]nodediscovery.SvcNode
var availableNodes map[string]nodediscovery.SvcNode
var availableNodesLock = sync.RWMutex{}
var lowestPing float32

// var active bool
var pingPeriod int

var pinger NodePinger

// var nodeLock sync.Mutex
var nodesToAdd map[string]nodediscovery.SvcNode
var nodeAddLock = sync.RWMutex{}

// var NodesUpdatedCallback func()
var nodeListeners map[string]string

func RegisterListener(name string, port string) {
	nodeListeners[name] = port
}

func GetKnownNodes() []nodediscovery.SvcNode {
	nodes := []nodediscovery.SvcNode{}

	availableNodesLock.Lock()
	for _, node := range availableNodes {
		if node.Distance != -1 {
			nodes = append(nodes, node)
		}
	}
	availableNodesLock.Unlock()
	return nodes
}

func StartNodeDiscoverySvc(ctx context.Context, p NodePinger, period int, nodes []nodediscovery.SvcNode) {
	go func() {
		StartDiscovery(ctx, p, period, nodes)
	}()
}

func StartDiscovery(ctx context.Context, p NodePinger, period int, nodes []nodediscovery.SvcNode) {
	//active = true
	nodeListeners = make(map[string]string)
	pinger = p
	pingPeriod = period
	lowestPing = 1000000
	nodesToAdd = make(map[string]nodediscovery.SvcNode)
	discoveredNodes = make(map[string]nodediscovery.SvcNode)
	availableNodes = make(map[string]nodediscovery.SvcNode)

	//nodes := make(map[string]string)
	//json.Unmarshal(nodejson, &nodes)
	for _, node := range nodes {
		discoveredNodes[node.Name] = node
	}

	transport := &http.Transport{
		MaxIdleConns:        150,
		MaxIdleConnsPerHost: 50,
		MaxConnsPerHost:     150,
		IdleConnTimeout:     30 * time.Second,
		//DisableCompression: true,
	}

	wsclient.HttpClient = http.Client{
		Transport: transport,
		Timeout:   time.Duration(2*config.Cfg.MaxPing) * time.Millisecond,
	}

	/*defer func() {
		if r := recover(); r != nil {
			log.Errorf("Recovered in NodeDiscovery %v", r)
		}
	}()*/

	//initialDiscover()
	for {
		select {
		case <-ctx.Done():
			//log.Info("Context canceled, stopping discovery")
			return
		default:
			start := time.Now()
			//log.Info("Ping/discover...")
			newNodes, deletedNodes := pingAndDiscover()

			availableNodesLock.Lock()
			availableNodes = make(map[string]nodediscovery.SvcNode)
			for key, val := range discoveredNodes {
				availableNodes[key] = val
			}
			availableNodesLock.Unlock()

			nodeList := []nodediscovery.SvcNode{}
			for _, node := range discoveredNodes {
				nodeList = append(nodeList, node)
			}

			for _, port := range nodeListeners {
				go func() {
					wsclient.NotifyNodesUpdatedListeners(port, nodeList)
				}()
				go func() {
					wsclient.NotifyNodesDiscoveredListeners(port, newNodes, deletedNodes)
				}()
			}
			timeMillis := time.Now().UnixMilli() - start.UnixMilli()
			sleepTime := int64(pingPeriod) - timeMillis
			log.Errorf("Discovery sleeping for %d ms", sleepTime)
			if sleepTime > 0 {
				time.Sleep(time.Duration(sleepTime) * time.Millisecond)
			}
		}
	}
}

func pingAndDiscover() ([]nodediscovery.SvcNode, []nodediscovery.SvcNode) {
	toPing := make(map[string]nodediscovery.SvcNode)
	tooFar := make(map[string]nodediscovery.SvcNode)
	newNodes := []nodediscovery.SvcNode{}
	removed := []nodediscovery.SvcNode{}
	//nodeLock.Lock()
	//defer nodeLock.Unlock()

	//discoveredNodesLock.Lock()
	for _, node := range discoveredNodes {
		toPing[node.Name] = node // = append(toPing, node)
	}
	//discoveredNodesLock.Unlock()

	//toPing = append(toPing, nodesToAdd...)
	nodeAddLock.Lock()
	for _, node := range nodesToAdd {
		_, contains := toPing[node.Name]
		if !contains {
			toPing[node.Name] = node // = append(toPing, node)
		}
	}
	/*for _, node := range nodesToAdd {
		toPing = append(toPing, node)
	}*/
	nodesToAdd = make(map[string]nodediscovery.SvcNode)
	nodeAddLock.Unlock()

	if len(toPing) > 0 {
		lowestPing, _ = wsclient.GetPing(toPing[getFirst(toPing)]) //toPing[0])
	}
	log.Info("Start discovery round")
	////log.Infof("Nodes 1 to ping %d", len(toPing))
	curNode := nodediscovery.SvcNode{}
	nodesWithinReach := false
	for len(toPing) > 0 {
		//curNode, toPing = toPing[0], toPing[1:]
		curNode = toPing[getFirst(toPing)]
		delete(toPing, curNode.Name)
		if curNode.IP != "" {
			if pinger.ShouldReping(curNode) {
				dist, err := wsclient.GetPing(curNode)
				if err == nil {
					curNode.Distance = dist

					//log.Infof("Node %s ping %f", curNode.Name, curNode.Distance)

					mPing := pinger.GetPingThreshold(len(discoveredNodes))
					//determine whether to add to "known" nodes or delete it based on ping
					if curNode.Distance <= mPing {
						//log.Debugf("Distance %f < %f, updating node", curNode.Distance, mPing)
						if !checkUpdateDiscoveredNode(curNode) {
							newNodes = append(newNodes, curNode)
						}

						newNodes2, _ := wsclient.GetKnownSvcNodes(curNode)
						toPing = mergeNodes(toPing, newNodes2, tooFar)
						nodesWithinReach = true

						if curNode.Distance < lowestPing {
							//log.Debugf("Lowest ping reduced to %f", lowestPing)
							lowestPing = curNode.Distance
						}
					} else {
						//log.Debugf("Distance %f > %f", curNode.Distance, mPing)
						tooFar[curNode.Name] = curNode //append(tooFar, curNode)
						//TODO: break the circle of pinging right here, don't let a node back in if we already removed it this ping round
						//if it's the last node remaining or has a lower ping than the others, let it stay anyway, it's the best link around
						if lowestPing > pinger.GetPingThreshold(len(discoveredNodes)) {
							//log.Info("Lowest ping higher than threshold, merging new nodes anyway")
							newNodes, _ := wsclient.GetKnownSvcNodes(curNode)
							toPing = mergeNodes(toPing, newNodes, tooFar)
						}
						if len(discoveredNodes) <= 1 || curNode.Distance <= lowestPing {
							//log.Debugf("Lowest ping reduced to %f", lowestPing)
							//log.Debugf("Not removing node %s due to last one or lowest ping", curNode.Name)
							lowestPing = curNode.Distance

							if !checkUpdateDiscoveredNode(curNode) {
								newNodes = append(newNodes, curNode)
							}
						} else {
							//discoveredNodesLock.Lock()
							_, contains := discoveredNodes[curNode.Name]
							if contains {
								//log.Debugf("Removing %s", curNode.Name)
								removed = append(removed, curNode)
								delete(discoveredNodes, curNode.Name)
							}
							//discoveredNodesLock.Unlock()
						}

					}
				}
			}
		}
	}

	if nodesWithinReach {
		//discoveredNodesLock.Lock()
		tooFar := []string{}
		for name, node := range discoveredNodes {
			if node.Distance > pinger.GetPingThreshold(len(discoveredNodes)) {
				tooFar = append(tooFar, name)
			}
		}

		for _, name := range tooFar {
			//log.Debugf("Post discovery removing %s because too far", name)
			removed = append(removed, discoveredNodes[name])
			delete(discoveredNodes, name)
		}
		//discoveredNodesLock.Unlock()
	}

	//jsonBytes, _ := json.Marshal(discoverMap)
	return newNodes, removed
}

func getFirst(nodes map[string]nodediscovery.SvcNode) string {
	for key, _ := range nodes {
		return key
	}
	return ""
}

func checkUpdateDiscoveredNode(curNode nodediscovery.SvcNode) bool {
	//discoveredNodesLock.Lock()
	_, contains := discoveredNodes[curNode.Name]
	discoveredNodes[curNode.Name] = curNode
	//discoveredNodesLock.Unlock()
	return contains
}

func mergeNodes(toPing map[string]nodediscovery.SvcNode, newNodes []nodediscovery.SvcNode, tooFar map[string]nodediscovery.SvcNode) map[string]nodediscovery.SvcNode {
	//log.Info("Merge nodes")
	//discoveredNodesLock.Lock()
	for _, node := range newNodes {
		_, discovered := discoveredNodes[node.Name]
		_, contains := toPing[node.Name] //containsElement(node, toPing)
		_, distant := tooFar[node.Name]  //containsElement(node, tooFar)
		//log.Debugf("Discoverednodes contains %s ? %t", node.Name, contains)
		if !contains && !discovered && !distant && node.Name != wsclient.GetNodeID() {
			//log.Debugf("Merging node %s", node.Name)
			toPing[node.Name] = node //append(toPing, node)
		}
	}
	//discoveredNodesLock.Unlock()
	return toPing
}

func containsElement(fn nodediscovery.SvcNode, array []nodediscovery.SvcNode) bool {
	contains := false
	for _, node := range array {
		if fn.Name == node.Name {
			contains = true
		}
	}

	return contains
}

func AddNode(node nodediscovery.Client, ip string) {
	//log.Debugf("Add node %s ip %s", node.Name, ip)
	if node.Name == wsclient.GetNodeID() {
		//log.Info("Won't add self, return")
		return
	}

	//nodeLock.Lock()
	//defer nodeLock.Unlock()
	//discoveredNodesLock.Lock()
	nodeAddLock.Lock()
	_, contains := nodesToAdd[node.Name]
	//discoveredNodesLock.Unlock()
	if !contains {
		//log.Debugf("Unknown node, creating and pinging")
		fogNode := nodediscovery.SvcNode{
			Name:         node.Name,
			IP:           ip,
			Distance:     -1,
			RepoAPIPort:  node.RepoAPIPort,
			DiscoAPIPort: node.DiscoAPIPort,
		}
		//fogNode.Distance = -1
		//discoveredNodes[name] = fogNode
		nodesToAdd[fogNode.Name] = fogNode // = append(nodesToAdd, fogNode)

		//log.Debugf("Added %s ip %s ping %f", node.Name, ip, float32(-1))
	} else {
		//log.Debugf("Node %s already known", node.Name)
	}
	nodeAddLock.Unlock()
}

type SwirlyNodePinger struct {
}

func (fp *SwirlyNodePinger) GetPingThreshold(nearbyNodes int) float32 {
	relativeDensity := 10 / float64(nearbyNodes)

	multi := float64(1)
	if relativeDensity > 1 {
		multi = math.Sqrt(relativeDensity)
	}
	//log.Debugf("Ping multiplier %f", multi)

	return config.Cfg.MaxPing * float32(multi)
}

func (fp *SwirlyNodePinger) ShouldReping(node nodediscovery.SvcNode) bool {
	return true
}
