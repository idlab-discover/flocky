# Flocky

This repo contains the source code for Flocky, a decentralized orchestrator based on Open Application Model using Kubernetes pods and OCI images as deployment units. 

## Flocky Overview

Flocky is a decentralized orchestrator based on [Open Application Model](https://github.com/oam-dev/spec). The main goal of the framework is to remote centralized bottlenecks by distributing orchestration responsibilities across an entire cluster instead of a control plane on one (or a few) nodes. This allows each node to discover other nodes, introduce new nodes and workloads into the cluster, and to find services offered by other nodes according to its needs.

The framework requires significant modeling capability to support highly heterogeneous edge environments. Compared to Kubernetes, OAM focuses more on application structure and workload behavior than detailing the technical minutiae of workload (containers). The base unit of deployment is an Application, which is composed of one or more Components. Components refer to ComponentDefinitions, which (unlike in Kubernetes Deployments) are declared separately to reduce the size and overhead in Application manifests. OAM also allows the definition of Traits and Scopes; Traits can be interpreted as intents, and may be applied to Components to ensure certain behavior or properties outside of the workload's logic (for example, setting up network properties or secure environments). Scopes may be used to group several Components into a single environment, similar to Kubernetes Pods but for specific operational aspects (not currently supported in Flocky). 

As OAM has a lot of freedom for implementation, Flocky uses Kubernetes Container definitions as the ComponentDefinition's workload spec. Furthermore, Flocky uses [Feather](https://github.com/togoetha/feather-multiruntimenetwork) as a workload agent, resulting in support for Docker containers, OSv unikernels, and WebAssembly (in development phase) workloads packaged as OCI images. Multiple images can be made for the same Component, by overloading ComponentDefinitions for each image and assigning them the same "BaseComponent" URI. The BaseComponent URI is then used in Application Components, and Flocky figures out which one to best use given the Traits to apply.

The image below shows the basic components (services) in Flocky. These are explored in great detail in the [accompanying article](https://www.researchgate.net/publication/391519467_Flocky_Decentralized_Intent-based_Edge_Orchestration_using_Open_Application_Model) (also published in IEEE Transactions on Service Computing). 

<img width="2116" height="1191" alt="architecture" src="https://github.com/user-attachments/assets/5f8eb744-b19e-4721-bf6f-9bf177c7d31d" />

### Core Services

The [**Discovery Service**](https://github.com/idlab-discover/flocky/tree/main/discoveryservice) is the basis of Flocky, responsible for finding other nodes and determining their position within the global cluster. Discovery must be smooth and fast, reacting as close to real-time as possible, so discovery is limited to nodes within a certain network latency, although higher-level services may impose different metrics (for example QoE). As a result, discovery performance depends only on local node density, no matter how big the entire cluster is. 

Configuration options (example in defaultconfig.json): 

- discoAPIPort: port at which the Discovery Service API is hosted.
- repoAPIPort: port at which the Repository Service is hosted.
- initialNodes: map of initially known nodes and their IP addresses. This map should be as limited as possible to encourage discovery; preferably only a single node from a collection of "introductory" nodes into the cluster.
- maxPing: maximum latency of nodes to discover. Nodes further away may be encountered (or even saved in extreme circumstances), but are discarded as soon as possible.
- nodeID: node name.
- pingPeriod: default period between attempts to update node statuses.
- testMode: very spicy, only to be used for evaluation runs as it overrides some of the normal operation of the service (for example, assuming incremental port numbers based on nodeID).

The **Repository Service** builds on the Discovery Service, and uses Flocky's OAM implementation to compose and maintain a repository of node metadata, known ComponentDefinitions, and services running on other nodes. While the Repository Service code is separate from the Discovery Service, it is located in the Discovery Service folder and these services are always run in the same executable since one is useless without the other. 

## Gossip Learning extensions

The image below shows the additions to Flocky, allowing it to be used for Gossip Learning. The services, along with improvements on Gossip Learning model integration itself, are detailed in a second paper ([preprint](https://arxiv.org/abs/2512.01549)).

<img width="1842" height="947" alt="architecture" src="https://github.com/user-attachments/assets/ef1594c1-e4da-4498-9c1d-760aa5412668" />

### Gossip Service

### ML Service

## Evaluation code & generators

### Base Flocky evaluation

### Gossip Learning evaluation

## Example deployment JSON of an Application using Gossip Learning

[exampleApp.json](exampleApp.json) contains an example of how to declare a multi-component Application which can be processed by the Swirly service. Notably, since implementation details (for example, the link to the OCI image and required resources) are declared by Component Implementations, deployment manifests are short and only specify Traits and concrete configuration info (e.g. runtime flags).
