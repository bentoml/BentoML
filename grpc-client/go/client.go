package main

import (
	"context"
	"fmt"
	"time"

	pb "github.com/bentoml/bentoml/grpc"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var opts []grpc.DialOption

const serverAddr = "localhost:3000"

func main() {
	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	conn, err := grpc.Dial(serverAddr, opts...)
	if err != nil {
		panic(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client := pb.NewBentoServiceClient(conn)

	req := &pb.Request{
		ApiName: "classify",
		Content: &pb.Request_Ndarray{
			Ndarray: &pb.NDArray{
				Dtype:       *pb.NDArray_DTYPE_FLOAT.Enum(),
				Shape:       []int32{1, 4},
				FloatValues: []float32{3.5, 2.4, 7.8, 5.1},
			},
		},
	}
	resp, err := client.Call(ctx, req)
	if err != nil {
		panic(err)
	}
	fmt.Print(resp)
}
