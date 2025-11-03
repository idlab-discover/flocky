package svcrepository

import (
	"context"
	"sync"

	"oamswirly/common/nodediscovery"
	"oamswirly/common/oam"
	"oamswirly/discoveryservice/config"
	"oamswirly/discoveryservice/svcrepository/local"
	"oamswirly/discoveryservice/svcrepository/wsclient"
	"time"

	log "github.com/sirupsen/logrus"
)

var discoveredComponentDefs map[string][]*oam.ComponentDef //index base component name
var discoveredComponentsLock = sync.RWMutex{}

// var runningApplications map[string]oam.NodeApps           //index node name or IP
var discoveredNodeCaps map[string]*oam.NodeSummary //index node name or IP

var knownNodes map[string]nodediscovery.SvcNode
var knownNodesMutex = sync.RWMutex{}

func AddNodes(nodes []nodediscovery.SvcNode) {
	for _, node := range nodes {
		knownNodesMutex.Lock()
		_, known := knownNodes[node.Name]
		if !known {
			knownNodes[node.Name] = node
			//ScanNode(node)
		}
		knownNodesMutex.Unlock()
	}
}

func DeleteNodes(nodes []nodediscovery.SvcNode) {
	for _, node := range nodes {
		knownNodesMutex.Lock()
		_, known := knownNodes[node.Name]
		if known {
			delete(knownNodes, node.Name)
			//delete(runningApplications, node.Name)
			delete(discoveredNodeCaps, node.Name)
		}
		knownNodesMutex.Unlock()
	}
}

// This is suspiciously like AddNodes, but it serves a different purpose and implementation may change
// it just so happens that the code is shortest and easiest this way atm
func UpdateNodeDistances(nodes []nodediscovery.SvcNode) {
	for _, node := range nodes {
		knownNodesMutex.Lock()
		_, known := knownNodes[node.Name]
		if known {
			knownNodes[node.Name] = node
		}
		knownNodesMutex.Unlock()
	}
}

func GetKnownComponentDefinitions() map[string][]string {
	defNames := make(map[string][]string)
	discoveredComponentsLock.Lock()
	for name, defs := range discoveredComponentDefs {
		names := []string{}
		for _, def := range defs {
			names = append(names, def.Metadata.Name)
		}
		defNames[name] = names
	}
	discoveredComponentsLock.Unlock()
	return defNames
}

func GetFullComponentDefinitions(comps []string) map[string][]*oam.ComponentDef {
	cDefs := make(map[string][]*oam.ComponentDef)
	discoveredComponentsLock.Lock()
	for _, comp := range comps {
		cDefs[comp] = discoveredComponentDefs[comp]
	}
	discoveredComponentsLock.Unlock()
	return cDefs
}

func GetKnownNodeSummaries() []*oam.NodeSummary {
	summaries := []*oam.NodeSummary{}
	for _, s := range discoveredNodeCaps {
		summaries = append(summaries, s)
	}
	return summaries
}

// This is a massive shortcut. Technically, trait "applies to workload" maps can differ per node depending on what other services
// they have running to supplement Flocky. However, testing multiple components PER node would make orchestration a massive clusterflock,
// so for now we're assuming each node's traits behave exactly the same if they have the same name.
// There's one gotcha: GetConcreteComponentsFor won't work unless a node supporting the requested Traits is within discovery distance. HAH.
// But then deployment wouldn't work either without such a node, so even steven.
func TraitDefsInRange() map[string]oam.TraitDef {
	distinctTraits := make(map[string]oam.TraitDef)

	for _, node := range GetKnownNodeSummaries() {
		for _, trait := range node.NodeCaps.Caps.SupportedTraits {
			distinctTraits[trait.Metadata.Name] = trait
		}
	}

	return distinctTraits
}

// We should be able to score the componentdefs by how well they match traits/user prefs later
func GetConcreteComponentsFor(cd oam.Component) map[int]*oam.ComponentDef {
	candidates := discoveredComponentDefs[cd.Type]

	reqTraits := []oam.Trait{}
	nonReqTraits := []oam.Trait{}
	for _, trait := range cd.Traits {
		if oam.IsRequiredTrait(trait) {
			reqTraits = append(reqTraits, trait)
		} else {
			nonReqTraits = append(nonReqTraits, trait)
		}
	}
	//log.Infof("GetConcreteComponentsFor %s %d required traits", cd.Name, len(reqTraits))
	traitDefs := TraitDefsInRange()
	acceptableDefs := []*oam.ComponentDef{}
	for _, cdef := range candidates {
		//log.Infof("Checking cDef %s", cdef.Metadata.Name)
		hasTraits := true
		for _, trait := range reqTraits {
			//cTrait := oam.TraitFromComponentInfo(trait)
			if !oam.CheckWorkloadTypeTrait(traitDefs[trait.Type], cdef.Spec.Workload, trait) {
				//log.Infof("Ignoring cDef, does not provide trait %s", trait.Type)
				hasTraits = false
			}
		}
		if hasTraits {
			//log.Infof("Adding acceptable cDef %s", cdef.Metadata.Name)
			acceptableDefs = append(acceptableDefs, cdef)
		}
	}

	//Score by non-required traits

	filteredDefs := make(map[int]*oam.ComponentDef)
	for i, cdef := range acceptableDefs {
		filteredDefs[i] = cdef
	}
	//log.Infof("Found %d acceptable cdefs for %s", len(acceptableDefs), cd.Name)
	return filteredDefs
}

func Start(ctx context.Context, bootstrap []nodediscovery.SvcNode, basicComponentDefs map[string][]*oam.ComponentDef) {
	discoveredComponentDefs = basicComponentDefs //make(map[string][]oam.ComponentDef)
	//runningApplications = make(map[string]oam.NodeApps)
	discoveredNodeCaps = make(map[string]*oam.NodeSummary)
	knownNodes = make(map[string]nodediscovery.SvcNode)
	for _, node := range bootstrap {
		knownNodes[node.Name] = node
	}
	wsclient.Init()
	wsclient.RegisterAsNodeListener(config.Cfg.DiscoAPIPort, config.Cfg.NodeID)
	local.Init()
	local.SyncLocalCapabilities(ctx)

	go func() {
		UpdateLoopLocalNodeCaps(ctx)
	}()
	go func() {
		ScanRemoteNodes(ctx)
	}()
}

func AddComponentDef(cDef *oam.ComponentDef) {
	discoveredComponentsLock.Lock()
	parentType := cDef.Metadata.Labels[oam.MetaConcreteComponentName]
	defs, found := discoveredComponentDefs[parentType]
	if found {
		defs = append(defs, cDef)
		discoveredComponentDefs[parentType] = defs
	} else {
		discoveredComponentDefs[parentType] = []*oam.ComponentDef{cDef}
	}
	discoveredComponentsLock.Unlock()
}

// Convert this to event-based i.e. register with a node to receive WL/caps updates? => if so, needs a timeout mechanism to prevent all nodes from eventually sending to all others
// Try it! but only for workloads/components, those are nice events. Resources change all the time, so even with eventing polling is required on the node itself -> same shit
// For traits and available runtimes, those rarely ever change and *could* be event based, but they're technically node resources and thus lumped in with volatile stuff
// Scrap that, instead of event based, going to make the entire thing timestamp based to just fetch latest changes
// However, timestamps are node dependent, so we'll have to keep the last one PER node
// And also, we're not doing that for now, because it's annoying to implement and debug, so let's do the important stuff first
func ScanRemoteNodes(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			//log.Info("Context canceled, stopping metadata repo")
			return
		default:
			start := time.Now()
			appsUpdates := []oam.NodeSummary{}
			capsUpdates := []oam.NodeSummary{}
			kNodes := make(map[string]nodediscovery.SvcNode)
			knownNodesMutex.Lock()
			for name, node := range knownNodes {
				kNodes[name] = node
			}
			knownNodesMutex.Unlock()

			for _, node := range kNodes {
				apps, caps, err := ScanNode(node)

				if err != nil {
					log.Errorf("ScanRemoteNodes error for node %s %v", node.Name, err)
				} else {
					if apps != nil {
						appsUpdates = append(appsUpdates, *apps)
					}
					capsUpdates = append(capsUpdates, *caps)
				}
			}
			wsclient.NotifyRemoteAppChangesListeners(appsUpdates)
			wsclient.NotifyRemoteCapsChangesListeners(capsUpdates)
			timeMillis := time.Now().UnixMilli() - start.UnixMilli()
			sleepTime := 5000 - timeMillis
			log.Errorf("Meta repo sleeping for %d ms", sleepTime)
			if sleepTime > 0 {
				time.Sleep(time.Millisecond * time.Duration(sleepTime))
			}
		}
	}
}

func ScanNode(node nodediscovery.SvcNode) (*oam.NodeSummary, *oam.NodeSummary, error) {
	//Get known component defs from remote node
	remoteKnownDefNames, _ := wsclient.GetKnownComponentDefs(node)
	missingNames := []string{}
	//Integrate
	//Due to multiple images per component definition depending on runtimes and specific traits required, each one actually has a bunch of "sub component defs"
	//Example: auth service -> container auth service, unikernel auth service, wasm auth service, ...
	for cDefType, subDefs := range remoteKnownDefNames {
		discoveredComponentsLock.Lock()
		defs, cDefTypeExists := discoveredComponentDefs[cDefType]
		if cDefTypeExists {
			//Shit well this is kind of a mess
			for _, newDef := range subDefs {
				found := false
				for _, oldDef := range defs {
					if oldDef.Metadata.Name == newDef {
						found = true
					}
				}
				if !found {
					missingNames = append(missingNames, cDefType)
				}
			}
			discoveredComponentDefs[cDefType] = defs
		} else {
			//discoveredComponentDefs[cDefType] = subDefs
			missingNames = append(missingNames, cDefType)
		}
		discoveredComponentsLock.Unlock()
	}

	if len(missingNames) > 0 {
		remoteKnownDefs, _ := wsclient.GetFullKnownComponentDefs(node, missingNames)
		for cDefType, subDefs := range remoteKnownDefs {
			discoveredComponentsLock.Lock()
			defs, cDefTypeExists := discoveredComponentDefs[cDefType]
			if cDefTypeExists {
				//Shit well this is kind of a mess
				for _, newDef := range subDefs {
					found := false
					for _, oldDef := range defs {
						if oldDef.Metadata.Name == newDef.Metadata.Name {
							found = true
						}
					}
					if !found {
						defs = append(defs, newDef)
					}
				}
				discoveredComponentDefs[cDefType] = defs
			} else {
				discoveredComponentDefs[cDefType] = subDefs
			}
			discoveredComponentsLock.Unlock()
		}
	}

	//Get node caps & running services summary
	remoteSummary, err := wsclient.GetNodeSummary(node)
	remoteSummary.NetInfo = oam.NetInfo{
		PrimaryAddress: node.IP,
		Latency:        node.Distance,
	}

	//Integrate
	//Based on last update we can do a diff and notify local listeners (i.e. the orchestration service to be) of remote changes
	lastUpdate := discoveredNodeCaps[node.Name]
	if lastUpdate == nil {
		discoveredNodeCaps[node.Name] = &oam.NodeSummary{}
		lastUpdate = &oam.NodeSummary{}
	}
	//runningApplications[node.Name] = remoteSummary.Applications
	if err != nil {
		remoteSummary = lastUpdate
	}
	newApps := oam.ApplicationArrayDiff(remoteSummary.NodeApps.Applications, lastUpdate.NodeApps.Applications)
	deletedApps := oam.ApplicationArrayDiff(lastUpdate.NodeApps.Applications, remoteSummary.NodeApps.Applications)

	var appsUpdate *oam.NodeSummary
	appsUpdate = nil
	if len(newApps) > 0 || len(deletedApps) > 0 {
		appsUpdate = &oam.NodeSummary{
			ApiVersion: "v1/beta",
			Kind:       "NodeSummary",
			Metadata:   oam.Metadata{},
			Name:       node.Name,
			NetInfo: oam.NetInfo{
				PrimaryAddress: node.IP,
				Latency:        node.Distance,
			},
			NodeApps: remoteSummary.NodeApps,
		}
	}

	discoveredNodeCaps[node.Name] = remoteSummary //.Capabilities
	capsUpdate := oam.NodeSummary{
		ApiVersion: "v1/beta",
		Kind:       "NodeSummary",
		Metadata:   oam.Metadata{},
		Name:       node.Name,
		NetInfo: oam.NetInfo{
			PrimaryAddress: node.IP,
			Latency:        node.Distance,
		},
		NodeCaps: remoteSummary.NodeCaps,
	}

	return appsUpdate, &capsUpdate, nil
}

// Even though this may work event based behind the scenes, just update it every 5 seconds in case too many requests arrive at once for the same info
// in which case, it's already buffered
func UpdateLoopLocalNodeCaps(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			//log.Info("Context canceled, stopping local syncing")
			return
		default:
			UpdateLocalNodeCaps()
			time.Sleep(time.Second * 5)
		}
	}
}

var localCaps oam.NodeCapsContent

func UpdateLocalNodeCaps() {
	caps, err := local.GetCapabilitiesSummary()
	if err != nil {
		log.Errorf("UpdateLocalNodeCaps error %v", err)
		return
	}
	localCaps = caps
}

func GetLocalNodeCaps() oam.NodeCapsContent {
	return localCaps
}
