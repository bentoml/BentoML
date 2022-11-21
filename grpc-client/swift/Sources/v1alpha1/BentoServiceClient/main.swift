#if compiler(>=5.6)
import Foundation
import GRPC
import grpc_client_swift_v1alpha1_grpc_lib
import NIOCore
import NIOPosix
import src_bentoml_grpc_v1alpha1_service_proto
import SwiftProtobuf

// Setup an `EventLoopGroup` for the connection to run on.
//
// See: https://github.com/apple/swift-nio#eventloops-and-eventloopgroups
let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)

var apiName: String = "classify"
var shape: [Int32] = [1, 4]
var data: [Float] = [3.5, 2.4, 7.8, 5.1]

// Make sure the group is shutdown when we're done with it.
defer {
  try! group.syncShutdownGracefully()
}

// Configure the channel, we're not using TLS so the connection is `insecure`.
let channel = try GRPCChannelPool.with(
  target: .host("localhost", port: 3000),
  transportSecurity: .plaintext,
  eventLoopGroup: group
)

// Close the connection when we're done with it.
defer {
  try! channel.close().wait()
}

// Provide the connection to the generated client.
let stubs = Bentoml_Grpc_V1alpha1_BentoServiceClient(channel: channel)

// Form the request with the NDArray, if one was provided.
let ndarray: Bentoml_Grpc_V1alpha1_NDArray = .with {
  $0.shape = shape
  $0.floatValues = data
  $0.dtype = Bentoml_Grpc_V1alpha1_NDArray.DType.float
}

let request: Bentoml_Grpc_V1alpha1_Request = .with {
  $0.apiName = apiName
  $0.ndarray = ndarray
}

let call = stubs.call(request)
do {
  let resp = try call.response.wait()
  if let content = resp.content {
    switch content {
    case let .ndarray(ndarray):
      print("Response: \(ndarray)")
    default:
      print("Currently only support NDArray response.")
    }
  }
} catch {
  print("Rpc failed \(try call.status.wait()): \(error)")
}
#else
@main
enum NotAvailable {
  static func main() {
    print("This example requires Swift >= 5.6")
  }
}
#endif // compiler(>=5.6)

