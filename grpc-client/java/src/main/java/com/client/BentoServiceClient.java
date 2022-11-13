package com.client;

import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.bentoml.grpc.v1.BentoServiceGrpc;
import com.bentoml.grpc.v1.BentoServiceGrpc.BentoServiceBlockingStub;
import com.bentoml.grpc.v1.BentoServiceGrpc.BentoServiceStub;
import com.bentoml.grpc.v1.NDArray;
import com.bentoml.grpc.v1.Request;
import com.bentoml.grpc.v1.RequestOrBuilder;
import com.bentoml.grpc.v1.Response;

public class BentoServiceClient {

  private static final Logger logger = Logger.getLogger(BentoServiceClient.class.getName());

  static Iterable<Integer> convert(int[] array) {
    return () -> Arrays.stream(array).iterator();
  }

  public static void main(String[] args) throws Exception {
    String apiName = "classify";
    int shape[] = { 1, 4 };
    Iterable<Integer> shapeIterable = convert(shape);
    Float array[] = { 3.5f, 2.4f, 7.8f, 5.1f };
    Iterable<Float> arrayIterable = Arrays.asList(array);
    // Access a service running on the local machine on port 50051
    String target = "localhost:3000";

    ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();
    try {
      BentoServiceBlockingStub blockingStub = BentoServiceGrpc.newBlockingStub(channel);

      NDArray.Builder builder = NDArray.newBuilder().addAllShape(shapeIterable).addAllFloatValues(arrayIterable).setDtype(NDArray.DType.DTYPE_FLOAT);

      Request req = Request.newBuilder().setApiName(apiName).setNdarray(builder).build();

      try {
        Response resp = blockingStub.call(req);
        Response.ContentCase contentCase = resp.getContentCase();
        if (contentCase != Response.ContentCase.NDARRAY) {
          throw new Exception("Currently only support NDArray response");
        }
        NDArray output = resp.getNdarray();
        logger.info("Response: " + resp.toString());
      } catch (StatusRuntimeException e) {
        logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
        return;
      }
    } finally {
      // ManagedChannels use resources like threads and TCP connections. To prevent
      // leaking these
      // resources the channel should be shut down when it will no longer be used. If
      // it may be used
      // again leave it running.
      channel.shutdownNow().awaitTermination(1, TimeUnit.SECONDS);
    }
  }
}
