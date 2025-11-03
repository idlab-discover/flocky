package orchestration

import (
	"fmt"
	"os/exec"

	log "github.com/sirupsen/logrus"
)

var Locator ComponentLocator

type ComponentLocator interface {
	Init() ComponentLocator
	UpdateComponentLocation(serviceName string, newIP string) error
}

type HostsComponentLocator struct {
}

func (sl *HostsComponentLocator) Init() ComponentLocator {
	return sl
}

func (sl *HostsComponentLocator) UpdateComponentLocation(componentName string, newIP string) error {
	err := removeHostsLine(componentName)
	if err != nil {
		return err
	}
	err = addHostsLine(componentName, newIP)
	return err
}

func removeHostsLine(host string) error {
	cmd := fmt.Sprintf("cat /etc/hosts | grep -w -v \"%s\" > /etc/hosts", host)
	_, err := ExecCmdBash(cmd)
	return err
}

func addHostsLine(host string, ip string) error {
	line := fmt.Sprintf("%s	%s", host, ip)
	cmd := fmt.Sprintf("echo \"%s\" >> /etc/hosts", line)
	_, err := ExecCmdBash(cmd)
	return err
}

func ExecCmdBash(dfCmd string) (string, error) {
	//log.Infof("Executing %s", dfCmd)
	cmd := exec.Command("sh", "-c", dfCmd)
	stdout, err := cmd.Output()

	if err != nil {
		log.Errorf("ExecCmdBash error %v", err.Error())
		return "", err
	}

	return string(stdout), nil
}
