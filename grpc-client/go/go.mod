module grpc_client_go

go 1.22.5

toolchain go1.23.0

require (
	github.com/bentoml/bentoml/grpc v0.0.0-unpublished
	google.golang.org/grpc v1.66.2
	google.golang.org/protobuf v1.34.2 // indirect
)

replace github.com/bentoml/bentoml/grpc v0.0.0-unpublished => ./github.com/bentoml/bentoml/grpc/v1

require (
	github.com/golang/protobuf v1.5.2 // indirect
	golang.org/x/net v0.29.0 // indirect
	golang.org/x/sys v0.25.0 // indirect
	golang.org/x/text v0.18.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20240903143218-8af14fe29dc1 // indirect
)
