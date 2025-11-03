module test/mmap

go 1.24.0

require (
	github.com/libopenstorage/gossip v0.0.1-rc1
	github.com/sirupsen/logrus v1.8.1
	golang.org/x/sys v0.32.0
)

require golang.org/x/sync v0.15.0 // indirect

replace (
	github.com/kubernetes-incubator/external-storage => github.com/libopenstorage/external-storage v0.20.4-openstorage-rc3
	k8s.io/api => k8s.io/api v0.21.6
	k8s.io/apiextensions-apiserver => k8s.io/apiextensions-apiserver v0.21.6
	k8s.io/apimachinery => k8s.io/apimachinery v0.21.6
	k8s.io/apiserver => k8s.io/apiserver v0.21.6
	k8s.io/cli-runtime => k8s.io/cli-runtime v0.21.6
	k8s.io/client-go => k8s.io/client-go v0.21.6
	k8s.io/cloud-provider => k8s.io/cloud-provider v0.21.6
	k8s.io/cluster-bootstrap => k8s.io/cluster-bootstrap v0.21.6
	k8s.io/code-generator => k8s.io/code-generator v0.21.6
	k8s.io/component-base => k8s.io/component-base v0.21.6
	k8s.io/component-helpers => k8s.io/component-helpers v0.21.6
	k8s.io/controller-manager => k8s.io/controller-manager v0.21.6
	k8s.io/cri-api => k8s.io/cri-api v0.21.6
	k8s.io/csi-translation-lib => k8s.io/csi-translation-lib v0.21.6
	k8s.io/kube-aggregator => k8s.io/kube-aggregator v0.21.6
	k8s.io/kube-controller-manager => k8s.io/kube-controller-manager v0.21.6
	k8s.io/kube-proxy => k8s.io/kube-proxy v0.21.6
	k8s.io/kube-scheduler => k8s.io/kube-scheduler v0.21.6
	k8s.io/kubectl => k8s.io/kubectl v0.21.6
	k8s.io/kubelet => k8s.io/kubelet v0.21.6
	k8s.io/legacy-cloud-providers => k8s.io/legacy-cloud-providers v0.21.6
	k8s.io/metrics => k8s.io/metrics v0.21.6
	k8s.io/mount-utils => k8s.io/mount-utils v0.21.6
	k8s.io/sample-apiserver => k8s.io/sample-apiserver v0.21.6
)

replace k8s.io/kubernetes => k8s.io/kubernetes v1.21.6

replace golang.org/x/exp/mmap => golang.org/x/exp/mmap v0.0.0-20250620022241

require (
	github.com/armon/go-metrics v0.3.3 // indirect
	github.com/google/btree v1.0.0 // indirect
	github.com/hashicorp/errwrap v1.0.0 // indirect
	github.com/hashicorp/go-immutable-radix v1.2.0 // indirect
	github.com/hashicorp/go-msgpack v0.5.5 // indirect
	github.com/hashicorp/go-multierror v1.0.0 // indirect
	github.com/hashicorp/go-sockaddr v1.0.2 // indirect
	github.com/hashicorp/golang-lru v0.5.4 // indirect
	github.com/hashicorp/logutils v1.0.0 // indirect
	github.com/hashicorp/memberlist v0.1.4 // indirect
	github.com/libopenstorage/openstorage/v10 v10.1.1 // indirect
	github.com/miekg/dns v1.1.35 // indirect
	github.com/sean-/seed v0.0.0-20170313163322-e2103e2c3529 // indirect
	golang.org/x/crypto v0.37.0 // indirect
	golang.org/x/net v0.39.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
)
