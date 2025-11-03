wget https://go.dev/dl/go1.23.5.linux-amd64.tar.gz
rm -rf /usr/local/go && tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

cd deploymentservice
go build -a -ldflags "-s -w" -o deploymentsvc *.go

cd ../discoveryservice
go build -a -ldflags "-s -w" -o discoverysvc *.go

cd ../dummyservices
go build -a -ldflags "-s -w" -o dummysvc *.go

cd ../monitoringapp
go build -a -ldflags "-s -w" -o monitoringapp *.go

cd ../runner
go build -a -ldflags "-s -w" -o runner *.go

cd ../swirlyservice
go build -a -ldflags "-s -w" -o swirlysvc *.go
