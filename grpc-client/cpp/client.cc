#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include "bentoml/grpc/v1/service.grpc.pb.h"
#include "bentoml/grpc/v1/service.pb.h"

using bentoml::grpc::v1::BentoService;
using bentoml::grpc::v1::NDArray;
using bentoml::grpc::v1::Request;
using bentoml::grpc::v1::Response;
using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::Status;

int main(int argc, char **argv) {
    auto stubs = BentoService::NewStub(grpc::CreateChannel(
        "localhost:3000", grpc::InsecureChannelCredentials()));
    std::vector<float> data = {3.5, 2.4, 7.8, 5.1};
    std::vector<int> shape = {1, 4};

    Request request;
    request.set_api_name("classify");

    NDArray *ndarray = request.mutable_ndarray();
    ndarray->mutable_shape()->Assign(shape.begin(), shape.end());
    ndarray->mutable_float_values()->Assign(data.begin(), data.end());

    Response resp;
    ClientContext context;

    // Storage for the status of the RPC upon completion.
    Status status = stubs->Call(&context, request, &resp);

    // Act upon the status of the actual RPC.
    if (!status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        return 1;
    }
    if (!resp.has_ndarray()) {
        std::cout << "Currently only accept output as NDArray." << std::endl;
        return 1;
    }
    std::cout << "response byte size: " << resp.ndarray().ByteSizeLong()
              << std::endl;
    return 0;
}
