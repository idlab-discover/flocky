package gossiping

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/gossipservice/oam"
	"time"

	log "github.com/sirupsen/logrus"
)

func StartRestListener(listener oam.RESTSettings) {
	go func() {
		for GossipActive {
			items := GetGossipForKey(listener.Key)
			if len(items.NodeData) > 0 {
				SendGossip(items, listener)
			}
			time.Sleep(time.Second * 3)
		}
	}()
}

func StartRestPoller(poll oam.RESTSettings) {
	go func() {
		for GossipActive {
			item, err := GetGossip(poll)
			if err == nil {
				PushGossipItem(GossipItem{
					Key:  poll.Key,
					Data: item,
				})
			}
			time.Sleep(time.Second * 3)
		}
	}()
}

func SendGossip(items GossipItems, endpoint oam.RESTSettings) error {
	clientJson, _ := json.Marshal(items)

	response, err := http.Post(endpoint.Endpoint, "application/json", bytes.NewBuffer(clientJson))

	if err != nil {
		log.Errorf("SendGossip error %v", err)
		return err
	}

	response.Body.Close()
	return nil
}

func GetGossip(endpoint oam.RESTSettings) ([]byte, error) {
	keyJson, _ := json.Marshal(endpoint.Key)
	//	endpoint := fmt.Sprintf("http://localhost:%d/%s", config.Cfg.DiscoAPIPort, "getKnownSvcNodes")
	resp, err := http.Post(endpoint.Endpoint, "application/json", bytes.NewBuffer(keyJson))
	if err != nil {
		return nil, fmt.Errorf("GetGossip %w", err)
	}
	defer resp.Body.Close()

	var content []byte
	if err := json.NewDecoder(resp.Body).Decode(&content); err != nil {
		return nil, fmt.Errorf("decode neighbors JSON: %w", err)
	}
	return content, nil
}
