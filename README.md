# Flocky

This repo contains the source code for Flocky, a decentralized orchestrator based on Open Application Model using Kubernetes pods and OCI images as deployment units. 

## Flocky Overview

The image below shows the basic components (services) in Flocky. These are explored in great detail in the [accompanying article](https://www.researchgate.net/publication/391519467_Flocky_Decentralized_Intent-based_Edge_Orchestration_using_Open_Application_Model) (also published in IEEE Transactions on Service Computing). 

<img width="2116" height="1191" alt="architecture" src="https://github.com/user-attachments/assets/5f8eb744-b19e-4721-bf6f-9bf177c7d31d" />

TODO explanation on OAM

### Core Services

TODO

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
