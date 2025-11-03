package gossiping

import (
	"oamswirly/gossipservice/oam"
	"oamswirly/gossipservice/sharedmem"
)

var openChannels []*sharedmem.ReadWriteMemoryChannel

func InitShm() {
	openChannels = []*sharedmem.ReadWriteMemoryChannel{}
}

func CleanupShm() {
	for _, channel := range openChannels {
		channel.Cleanup()
	}
}

func StartShmPoller(poll oam.ShmSettings) {
	go func() {
		channel := &sharedmem.ReadWriteMemoryChannel{}
		openChannels = append(openChannels, channel)
		channel.Init(poll.ChannelPath, poll.SemaWait, poll.SemaSignal, poll.ChannelName, int64(poll.Size), false)
		channel.StartRead(MakeReadWeights(poll.Key, poll.Size))
	}()
}

func MakeReadWeights(key string, size int) func([]byte) {
	return func(data []byte) {
		dataCopy := make([]byte, size)
		copy(dataCopy, data)

		PushGossipItem(GossipItem{
			Key:  key,
			Data: dataCopy,
		})
	}
}

func StartShmWriter(listener oam.ShmSettings) {
	go func() {
		channel := &sharedmem.ReadWriteMemoryChannel{}
		openChannels = append(openChannels, channel)
		channel.Init(listener.ChannelPath, listener.SemaWait, listener.SemaSignal, listener.ChannelName, int64(listener.Size), true)
		channel.StartWrite(MakeWriteWeights(listener.Key, listener.Size))
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
