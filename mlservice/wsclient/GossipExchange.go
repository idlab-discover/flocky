package wsclient

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/gossipservice/config"
	"oamswirly/gossipservice/gossiping"
	goam "oamswirly/gossipservice/oam"
	mloam "oamswirly/mlservice/oam"
	"oamswirly/mlservice/sharedmem"

	log "github.com/sirupsen/logrus"
)

func SetupGossipToMLLink(linkInfo oam.Trait) {
	if settings := mloam.GetReadShmSettings(linkInfo); len(settings) > 0 {
		SetupShmLink(linkInfo)
	} else if settings := mloam.GetReadRESTSettings(linkInfo); len(settings) > 0 {
		SetupRESTLink(linkInfo)
	} else {
		//Well they said "set up the link", but there's nothing to link to, so set up nothing.
	}
}

func ModelUpdatesReceived(model string, nodeData map[string][]byte) {

}

func SetupShmLink(linkInfo oam.Trait) {
	settings := mloam.GetReadShmSettings(linkInfo)
	//Only one read shm and one write shm supported for now.. these things are highly unstable and annoying to handle
	StartShmPoller(settings[0])

	settings = mloam.GetWriteShmSettings(linkInfo)
	StartShmWriter(settings[0])
}

func SetupRESTLink(linkInfo oam.Trait) {
	//Not supported atm
}

var poller *sharedmem.ReadWriteMemoryChannel
var writer *sharedmem.ReadWriteMemoryChannel

func Cleanup() {
	poller.Cleanup()
	writer.Cleanup()
}

func StartShmPoller(poll mloam.ShmSettings) {
	go func() {
		poller = &sharedmem.ReadWriteMemoryChannel{}
		poller.Init(poll.ChannelPath, poll.SemaWait, poll.SemaSignal, poll.ChannelName, int64(poll.Size), false)
		poller.StartRead(MakeReadWeights(poll.Key, poll.Size))
	}()
}

func MakeReadWeights(key string, size int) func([]byte) {
	return func(data []byte) {
		dataCopy := make([]byte, size)
		copy(dataCopy, data)

		PushGossipItem(gossiping.GossipItem{
			Key:  key,
			Data: dataCopy,
		})
	}
}

func PushGossipItem(item gossiping.GossipItem) {
	svc, _ := GetGossipService()

	clientJson, _ := json.Marshal(item)

	url := fmt.Sprintf("http://localhost:%d/pushGossipItem", svc.LocalEndpoint)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("PushGossipItem error %v", err)
		return
	}

	response.Body.Close()
}

func StartShmWriter(listener mloam.ShmSettings) {
	go func() {
		writer = &sharedmem.ReadWriteMemoryChannel{}
		writer.Init(listener.ChannelPath, listener.SemaWait, listener.SemaSignal, listener.ChannelName, int64(listener.Size), true)
		writer.StartWrite(MakeWriteWeights(listener.Key, listener.Size))
	}()
}

func MakeWriteWeights(key string, size int) func() [][]byte {
	return func() [][]byte {
		items := GetGossipForKey(key)

		serialized := [][]byte{}
		for _, item := range items.NodeData {
			itemCopy := make([]byte, size)
			copy(item, itemCopy)

			serialized = append(serialized, item)
		}
		return serialized
	}
}

func GetGossipForKey(key string) gossiping.GossipItems {
	svc, _ := GetGossipService()

	clientJson, _ := json.Marshal(key)

	url := fmt.Sprintf("http://localhost:%d/getGossipItems", svc.LocalEndpoint)
	//log.Infof("Registering caps provider %s for %s", url, string(clientJson))
	response, err := http.Post(url, "application/json", bytes.NewBuffer(clientJson))

	items := gossiping.GossipItems{}
	err = json.NewDecoder(response.Body).Decode(&items)
	if err != nil {
		log.Errorf("GetFlockyService JSON error %v", err)
	}

	response.Body.Close()
	return items
}

var gossipSvc oam.ServiceProvider = oam.ServiceProvider{}

func GetGossipService() (oam.ServiceProvider, error) {
	//_, cached := flockyServices[svc]
	svc := goam.SvcName

	if gossipSvc.LocalEndpoint == "" {
		jsonBytes, err := json.Marshal(svc)
		response, err := http.Post(fmt.Sprintf("http://localhost:%d/getFlockyServiceEndpoint", config.Cfg.RepoAPIPort), "application/json", bytes.NewBuffer(jsonBytes))
		if err != nil {
			log.Errorf("GetGossipService error %v", err)
			return oam.ServiceProvider{}, err
		}

		provider := oam.ServiceProvider{}
		err = json.NewDecoder(response.Body).Decode(&provider)
		if err != nil {
			log.Errorf("GetGossipService JSON error %v", err)
			return provider, err
		}

		response.Body.Close()
		gossipSvc = provider
	}

	return gossipSvc, nil
}
