Here you can see what the modified files of the gossip library are changed and what is changed or added

In the api.go file the Gossiper interface type changed and the Join(peers []string) (int, error) function was added
api.go:
```go
type Gossiper interface {
    // Gossiper has a gossip store
    GossipStore
    
    // Start begins the gossip protocol using memberlist
    // To join an existing cluster provide atleast one ip of the known node.
    Start(startConfiguration types.GossipStartConfiguration) error

    // GossipInterval gets the gossip interval
    GossipInterval() time.Duration

    // Join attempts to connect this node to the given list of peer addresses.
    Join(peers []string) (int, error)

    // Stop stops the gossiping. Leave timeout indicates the minimum time
    // required to successfully broadcast the leave message to all other nodes.
    Stop(leaveTimeout time.Duration) error

    // GetNodes returns a list of the connection addresses
    GetNodes() []string

    // UpdateCluster updates gossip with latest peer nodes info
    UpdateCluster(map[types.NodeId]types.NodeUpdate)
  
    // ExternalNodeLeave is used to indicate gossip that one of the nodes might be down.
    // It checks quorum and appropriately marks either self down or the other node down.
    // It returns the nodeId that was marked down
    ExternalNodeLeave(nodeId types.NodeId) types.NodeId

    // UpdateClusterDomainsActiveMap updates the cluster domain active map
    // All the nodes in an inactive domain will shoot themselves down
    // and will not participate in quorum decisions
    UpdateClusterDomainsActiveMap(types.ClusterDomainsActiveMap) error

    // UpdateSelfClusterDomain updates this node's cluster domain
    UpdateSelfClusterDomain(selfFailureDomain string)

    // Ping pings the given node's ip:port
    // Note: This API is only supported with Gossip Version v2 and higher
    Ping(nodeId types.NodeId, ipPort string) (time.Duration, error)
}
```

In the /proto/gossip.go file in the lirbary the Join function was introduced.
/proto/gossip.go:
```go
func (g *GossiperImpl) Join(peers []string) (int, error) {
    g.joinLock.Lock()
    defer g.joinLock.Unlock()
    if g.mlist == nil {
        return 0, fmt.Errorf("gossip: cannot join, memberlist not started")
    }

    // memberlist.Join returns the number of nodes successfully contacted
    return g.mlist.Join(peers)
}
```

