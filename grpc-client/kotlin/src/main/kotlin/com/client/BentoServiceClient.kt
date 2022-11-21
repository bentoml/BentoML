package com.client

import com.bentoml.grpc.v1.BentoServiceGrpc
import com.bentoml.grpc.v1.NDArray
import com.bentoml.grpc.v1.Request
import io.grpc.ManagedChannelBuilder

class BentoServiceClient {
  companion object {
    @JvmStatic
    fun main(args: Array<String>) {
      val apiName: String = "classify"
      val shape: List<Int> = listOf(1, 4)
      val data: List<Float> = listOf(3.5f, 2.4f, 7.8f, 5.1f)

      val channel = ManagedChannelBuilder.forAddress("localhost", 3000).usePlaintext().build()

      val client = BentoServiceGrpc.newBlockingStub(channel)

      val ndarray = NDArray.newBuilder().addAllShape(shape).addAllFloatValues(data).build()
      val req = Request.newBuilder().setApiName(apiName).setNdarray(ndarray).build()
      try {
        val resp = client.call(req)
        if (!resp.hasNdarray()) {
          println("Currently only support NDArray response.")
        } else {
          println("Response: ${resp.ndarray}")
        }
      } catch (e: Exception) {
        println("Rpc error: ${e.message}")
      }
    }
  }
}
