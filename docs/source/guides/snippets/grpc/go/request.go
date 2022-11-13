package main

import (
	pb "github.com/bentoml/bentoml/grpc/v1"
)

var req = &pb.Request{
	ApiName: "classify",
	Content: &pb.Request_Ndarray{
		Ndarray: &pb.NDArray{
			Dtype:       *pb.NDArray_DTYPE_FLOAT.Enum(),
			Shape:       []int32{1, 4},
			FloatValues: []float32{3.5, 2.4, 7.8, 5.1},
		},
	},
}
