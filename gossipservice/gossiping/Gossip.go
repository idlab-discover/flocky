package gossiping

import (
	"encoding/json"
	"fmt"
	"oamswirly/gossipservice/config"
	"oamswirly/gossipservice/wsclient"
	"strconv"
	"strings"
	"time"

	"github.com/libopenstorage/gossip"
	"github.com/libopenstorage/gossip/types"
	log "github.com/sirupsen/logrus"
)

var selfNodeID types.NodeId
var lastProcessedGossipTimestamp map[types.NodeId]time.Time
var gossiper gossip.Gossiper
var neighbour_count int

type GossipItems struct {
	Key      string
	NodeData map[string][]byte
}

type GossipItem struct {
	Key  string
	Data []byte
}

var GossipActive bool

func Cleanup() {
	gossiper.Stop(5 * time.Second)
	GossipActive = false
}

func InitGossiping() {
	//-------------------------------------------------------
	// Get initial neighbours
	//-------------------------------------------------------
	log.Info("Getting initial neighbors from discovery service")
	initialPeers := createInitialPeers()

	//-------------------------------------------------------
	// Initialize gossip info
	// Make gossiper
	//-------------------------------------------------------

	selfNodeID = types.NodeId(config.Cfg.NodeID)
	port := config.Cfg.ClusterStartPort

	gossiper = createGossiper(selfNodeID, port, initialPeers)
	GossipActive = true
	//-------------------------------------------------------
	// Go routines for neighbour initialization to join an existing cluster
	// And add new neighbours as it runs
	//-------------------------------------------------------

	go func() {
		updatePeersFromDiscovery(gossiper, selfNodeID)
	}()
}

func createInitialPeers() map[types.NodeId]types.GossipNodeConfiguration {

	initialPeers := make(map[types.NodeId]types.GossipNodeConfiguration, 0)

	if config.Cfg.NodeID != "f0" {
		/*neighbours, err := ws.LoadNeighbours()
		if err != nil {
			log.Fatalf("Getting neighbours error: %v", err)
		}*/
		neighbours := []wsclient.SvcNode{}
		for {
			neighbours, err := wsclient.LoadNeighbours()
			if err != nil {
				log.Info("Error getting neighbours, retrying...")
				time.Sleep(1 * time.Second)
				continue
			}
			if len(neighbours) == 0 {
				log.Infof("No neighbours yet, retrying discovery in 1s…")
				time.Sleep(1 * time.Second)
				continue
			} else {
				break
			}
		}

		log.Infof("Gotten neigbours: %v", neighbours)

		initialPeers = make(map[types.NodeId]types.GossipNodeConfiguration, len(neighbours))
		for _, svc := range neighbours {
			id := types.NodeId(svc.Name)                                // cast string → NodeId
			url := fmt.Sprintf("%s:%d", svc.IP, svc.DiscoAPIPort-22000) // "10.0.0.2:8000"
			initialPeers[id] = types.GossipNodeConfiguration{
				KnownUrl:      url,
				ClusterDomain: "", // or fill from svc if you have domains
			}
		}
	}
	//neighbour_count = len(initialPeers)
	return initialPeers
}

func createGossiper(selfNodeID types.NodeId, port int, initialPeers map[types.NodeId]types.GossipNodeConfiguration) gossip.Gossiper {
	ip := strings.Join([]string{"127.0.0.1", strconv.Itoa(port)}, ":")
	// ip := fmt.Sprintf("127.0.0.1:%s", port)
	log.Infof("[GOSSIPER] Gossip IP: %s", ip)
	gossipIntervals := types.GossipIntervals{
		GossipInterval:   1 * time.Second,
		PushPullInterval: 1 * time.Second,
		ProbeInterval:    1 * time.Second,
		ProbeTimeout:     500 * time.Millisecond,
	}

	gossiper := gossip.New(
		ip,
		selfNodeID,
		0,
		gossipIntervals,
		"v1",
		"cluster1",
		"",
	)

	log.Info("[GOSSIPER] gossiper Meta info", gossiper.MetaInfo())

	startCfg := types.GossipStartConfiguration{
		Nodes:              initialPeers,
		ActiveMap:          types.ClusterDomainsActiveMap{"": types.CLUSTER_DOMAIN_STATE_ACTIVE},
		QuorumProviderType: types.QUORUM_PROVIDER_DEFAULT,
	}
	log.Infof("Starting gossip with config: %v", startCfg)
	// Attempt to start once; ignore join errors
	if err := gossiper.Start(startCfg); err != nil {
		log.Warnf("Initial join failed: %v", err)
	}
	gossiper.UpdateSelfStatus(types.NODE_STATUS_UP)
	return gossiper
}

func updatePeersFromDiscovery(gossiper gossip.Gossiper, selfNodeID types.NodeId) {
	// Keep track of the last set of peers that was successfully sent to UpdateCluster.
	// This helps in determining if the current set from discovery has changed.
	lastPeersSentToUpdate := make(map[types.NodeId]types.NodeUpdate)
	// Keep track of peer IDs that have been logged as "Discovered peer" at least once, for logging purposes.
	loggedAsDiscoveredPeerIDs := make(map[types.NodeId]struct{})

	ticker := time.NewTicker(5 * time.Second) // retry interval
	defer ticker.Stop()
	for range ticker.C {
		svcs, err := wsclient.LoadNeighbours()
		neighbour_count = len(lastPeersSentToUpdate)
		if err != nil {
			log.Warnf("Discovery error: %v", err)
			continue
		}

		// Build the map of all peers currently found by the discovery service.
		currentPeersFromDiscovery := make(map[types.NodeId]types.NodeUpdate)
		for _, svc := range svcs {
			peerID := types.NodeId(svc.Name)

			// Skip selfNodeID if it appears in the list.
			if peerID == selfNodeID {
				continue
			}

			// Log if it's the first time this discovery loop encounters this peer.
			if _, exists := loggedAsDiscoveredPeerIDs[peerID]; !exists {
				log.Infof("Discovered peer via service discovery: %s. It will be part of the next cluster state evaluation.", peerID)
				loggedAsDiscoveredPeerIDs[peerID] = struct{}{}
			}

			currentPeersFromDiscovery[peerID] = types.NodeUpdate{
				Addr:          fmt.Sprintf("%s:%d", svc.IP, svc.DiscoAPIPort-22000),
				QuorumMember:  true, // Assuming new peers should be quorum members
				ClusterDomain: "",   // Assuming default cluster domain
			}
		}

		currentPeersFromDiscovery[selfNodeID] = types.NodeUpdate{
			Addr:          fmt.Sprintf("%s:%d", "127.0.0.1", config.Cfg.ClusterStartPort),
			QuorumMember:  true, // Assuming new peers should be quorum members
			ClusterDomain: "",   // Assuming default cluster domain
		}

		// Determine if the set of peers (or their details) has changed since the last update.
		hasChanged := false
		if len(currentPeersFromDiscovery) != len(lastPeersSentToUpdate) {
			hasChanged = true
		} else {
			// Lengths are the same, check for content differences.
			for peerID, currentData := range currentPeersFromDiscovery {
				lastData, ok := lastPeersSentToUpdate[peerID]
				// If peer was not in the old map, or its data changed.
				if !ok || currentData.Addr != lastData.Addr ||
					currentData.QuorumMember != lastData.QuorumMember ||
					currentData.ClusterDomain != lastData.ClusterDomain {
					hasChanged = true
					break
				}
			}
		}

		if hasChanged {
			urls := []string{}
			for _, peer := range currentPeersFromDiscovery {
				if !strings.HasSuffix(peer.Addr, fmt.Sprintf("%d", config.Cfg.ClusterStartPort)) { //"127.0.0.1") {
					urls = append(urls, peer.Addr)
				}
			}
			/*
				If missing "Join" method in default library:
				func (g *GossiperImpl) Join(peers []string) (int, error) {
					g.joinLock.Lock()
					defer g.joinLock.Unlock()
					if g.mlist == nil {
						return 0, fmt.Errorf("gossip: cannot join, memberlist not started")
					}
					return g.mlist.Join(peers)
				}
			*/
			if n, err := gossiper.Join(urls); err != nil {
				log.Warnf("Batch join failed: %v", err)
			} else {
				log.Infof("Successfully joined %d peer(s)", n)
			}

			log.Infof("Peer set from discovery has changed. Updating gossip cluster with %d peers: %v", len(currentPeersFromDiscovery), currentPeersFromDiscovery)
			gossiper.UpdateCluster(currentPeersFromDiscovery)

			// Update the record of the last set of peers sent to UpdateCluster.
			// Create a new map to avoid issues with map modifications if currentPeersFromDiscovery is reused.
			newLastPeersSentToUpdate := make(map[types.NodeId]types.NodeUpdate, len(currentPeersFromDiscovery))
			for k, v := range currentPeersFromDiscovery {
				newLastPeersSentToUpdate[k] = v
			}
			lastPeersSentToUpdate = newLastPeersSentToUpdate
		} else {
			log.Debugf("Peer set from discovery has not changed. No update to gossip library. Current set size: %d", len(currentPeersFromDiscovery))
		}
	}
}

type NodeData struct {
	Timestamp time.Time `json:"timestamp"`
	Data      []byte    `json:"data"`
}

func GetGossipForKey(key string) GossipItems {
	log.Info("GetGossipForKey")
	items := GossipItems{Key: key}

	nodeInfoMap := gossiper.GetStoreKeyValue(types.StoreKey(key))
	log.Infof("Checking store key %s", key)
	for nodeID, nodeInfo := range nodeInfoMap {
		log.Infof("Checking node %s status %s", nodeID, string(nodeInfo.Status))
		if nodeID != selfNodeID &&
			nodeInfo.Status != types.NODE_STATUS_INVALID { //&&
			//nodeInfo.Status == types.NODE_STATUS_UP {
			if nodeInfo.Value != nil {
				var receivedData NodeData
				if err := json.Unmarshal(nodeInfo.Value.([]byte), &receivedData); err != nil {
					log.Warningf("Failed to unmarshal data: %v", err)
					continue
				}
				// Check if we've already processed this version of data or if it's newer
				lastTimestamp, exists := lastProcessedGossipTimestamp[nodeID]
				if !exists || receivedData.Timestamp.After(lastTimestamp) {
					log.Infof("Processing new matrix from %s at %v (Previous timestamp: %v, Existed: %v)",
						nodeID, receivedData.Timestamp, lastTimestamp, exists)

					log.Infof("Processing new matrix from %s at %v (Previous timestamp: %v, Existed: %v)",
						nodeID, receivedData.Timestamp, lastTimestamp, exists)

					items.NodeData[string(nodeID)] = receivedData.Data
					lastProcessedGossipTimestamp[nodeID] = receivedData.Timestamp // Update last processed timestamp
				} else {
					log.Debugf("Skipping already processed or older data from node %s (Received: %v, LastProcessed: %v)",
						nodeID, receivedData.Timestamp, lastTimestamp)
				}
			} else {
				log.Infof("No update from %s", nodeID)
			}
		}
	}
	//}
	return items
}

func PushGossipItem(item GossipItem) {
	var nodeData NodeData

	//utils.LogWeightDetails(latestW.BaseWeights, "from SHM", sharedmem.Metadata)

	nodeData.Timestamp = time.Now()
	nodeData.Data = item.Data
	jsonData, err := json.Marshal(nodeData)
	if err != nil {
		log.Fatalf("Failed to marshal data: %v", err)
	}
	gossiper.UpdateSelf(types.StoreKey(item.Key), jsonData)

	log.Info("Applied latest data for key %s", item.Key)
}
