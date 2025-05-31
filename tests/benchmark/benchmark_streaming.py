import asyncio
import time
import statistics
import argparse
import httpx
import grpc # For status codes
from typing import List, Dict, Any, AsyncIterator

# Assuming the client is in the BentoML package and accessible
# Adjust the import path if necessary based on your project structure
from bentoml.grpc.v1alpha1.client import BentoMlGrpcClient
from bentoml.grpc.v1alpha1 import bentoml_service_v1alpha1_pb2 as pb  # For gRPC Request/Response types

# --- Constants ---
DEFAULT_ITERATIONS = 10
DEFAULT_GRPC_PORT = 50051
DEFAULT_REST_PORT = 3000 # Commonly used by BentoML REST servers
DEFAULT_HOST = "localhost"

# --- Helper Functions ---
def generate_payload(size: int) -> str:
    """Generates a string payload of a given size."""
    return "a" * size

async def safe_close_grpc_client(client: BentoMlGrpcClient | None):
    if client:
        await client.close()

# --- gRPC Benchmark ---
async def benchmark_grpc_stream(
    host: str,
    port: int,
    payload_size: int,
    stream_length: int,
    iterations: int
) -> Dict[str, Any]:
    """Benchmarks gRPC CallStream."""
    client = None
    timings = []
    first_response_latencies = []
    total_bytes_streamed_list = []
    successful_iterations = 0

    payload = generate_payload(payload_size)
    print(f"gRPC: Payload size: {payload_size} bytes, Stream length: {stream_length} messages, Iterations: {iterations}")

    try:
        client = BentoMlGrpcClient(host=host, port=port)
        # Warm-up call (optional, but good for stable measurements)
        async for _ in client.call_stream(data="warmup"):
            pass

        for i in range(iterations):
            start_time = time.perf_counter()
            first_response_time = None
            
            message_count = 0
            current_iteration_bytes = 0
            try:
                async for response in client.call_stream(data=payload):
                    if first_response_time is None:
                        first_response_time = time.perf_counter()
                    message_count += 1
                    current_iteration_bytes += len(response.data.encode('utf-8'))
                    if message_count >= stream_length: # Ensure we don't stream indefinitely if server sends more
                        break
                end_time = time.perf_counter()

                if message_count == stream_length:
                    timings.append(end_time - start_time)
                    if first_response_time:
                        first_response_latencies.append(first_response_time - start_time)
                    total_bytes_streamed_list.append(current_iteration_bytes)
                    successful_iterations +=1
                else:
                    print(f"gRPC Iteration {i+1}: Failed - Expected {stream_length} messages, got {message_count}")

            except grpc.aio.AioRpcError as e:
                print(f"gRPC Iteration {i+1}: Failed with gRPC error - {e.code()}: {e.details()}")
            except Exception as e:
                print(f"gRPC Iteration {i+1}: Failed with error - {e}")
            await asyncio.sleep(0.01) # Small delay between iterations

    finally:
        await safe_close_grpc_client(client)

    if not timings:
        return {"error": "No successful gRPC iterations."}

    avg_time = statistics.mean(timings)
    std_dev_time = statistics.stdev(timings) if len(timings) > 1 else 0
    avg_first_response_latency = statistics.mean(first_response_latencies) if first_response_latencies else -1
    
    total_bytes_per_iteration = statistics.mean(total_bytes_streamed_list)
    throughput_mps = successful_iterations / sum(timings) if sum(timings) > 0 else 0 # Total successful messages / total time for successful
    throughput_bps = sum(total_bytes_streamed_list) / sum(timings) if sum(timings) > 0 else 0


    return {
        "successful_iterations": successful_iterations,
        "avg_stream_time_s": avg_time,
        "std_dev_stream_time_s": std_dev_time,
        "avg_first_response_latency_s": avg_first_response_latency,
        "throughput_streams_per_s": 1 / avg_time if avg_time > 0 else 0,
        "throughput_msgs_per_s": throughput_mps * stream_length, # average messages per second across successful streams
        "throughput_bytes_per_s": throughput_bps,
        "avg_bytes_per_stream": total_bytes_per_iteration,
    }

# --- REST (HTTP/1.1 Streaming) Benchmark ---
async def benchmark_rest_stream(
    host: str,
    port: int,
    payload_size: int,
    stream_length: int,
    iterations: int,
    endpoint: str = "/stream" # Assuming a /stream endpoint for REST
) -> Dict[str, Any]:
    """Benchmarks REST streaming (e.g., line-delimited JSON or SSE)."""
    timings = []
    first_response_latencies = []
    total_bytes_streamed_list = []
    successful_iterations = 0

    payload_str = generate_payload(payload_size)
    # For REST, we'd typically send JSON. The server would need to handle this.
    # The 'data' field matches the gRPC Request message.
    json_payload = {"data": payload_str} 
    # We also need to inform the server about the desired stream length,
    # as HTTP streaming doesn't inherently have a message count like gRPC stream definition.
    # This could be a header or part of the JSON payload.
    # For this example, let's assume the server knows to send `stream_length` messages
    # or we pass it as a query parameter or in the payload.
    # Let's add it to the JSON payload for simplicity here.
    json_payload_with_length = {"data": payload_str, "stream_length": stream_length}

    print(f"REST: Payload size: {payload_size} bytes, Stream length: {stream_length} messages, Iterations: {iterations}")

    async with httpx.AsyncClient(base_url=f"http://{host}:{port}") as client:
        # Warm-up call
        try:
            async with client.stream("POST", endpoint, json={"data": "warmup", "stream_length": 1}) as response:
                async for _ in response.aiter_lines():
                    pass
        except httpx.RequestError as e:
            print(f"REST Warmup failed: {e}. Ensure REST server is running and endpoint '{endpoint}' exists.")
            # return {"error": f"Warmup failed: {e}"} # Or decide to continue

        for i in range(iterations):
            start_time = time.perf_counter()
            first_response_time = None
            message_count = 0
            current_iteration_bytes = 0
            try:
                async with client.stream("POST", endpoint, json=json_payload_with_length, timeout=30.0) as response:
                    if response.status_code != 200:
                        print(f"REST Iteration {i+1}: Failed - Status {response.status_code}, {await response.aread()}")
                        continue
                    
                    # Assuming server sends line-delimited text, each line is a "message"
                    async for line in response.aiter_lines():
                        if first_response_time is None:
                            first_response_time = time.perf_counter()
                        message_count += 1
                        current_iteration_bytes += len(line.encode('utf-8'))
                        # No break here, assume server sends exactly stream_length messages based on input
                end_time = time.perf_counter()

                if message_count == stream_length: # Validate if server sent expected number of messages
                    timings.append(end_time - start_time)
                    if first_response_time:
                        first_response_latencies.append(first_response_time - start_time)
                    total_bytes_streamed_list.append(current_iteration_bytes)
                    successful_iterations += 1
                else:
                    print(f"REST Iteration {i+1}: Failed - Expected {stream_length} messages, got {message_count}")

            except httpx.RequestError as e:
                print(f"REST Iteration {i+1}: Request failed - {e}")
            except Exception as e:
                print(f"REST Iteration {i+1}: Failed with error - {e}")
            await asyncio.sleep(0.01) # Small delay

    if not timings:
        return {"error": "No successful REST iterations."}

    avg_time = statistics.mean(timings)
    std_dev_time = statistics.stdev(timings) if len(timings) > 1 else 0
    avg_first_response_latency = statistics.mean(first_response_latencies) if first_response_latencies else -1

    total_bytes_per_iteration = statistics.mean(total_bytes_streamed_list)
    throughput_mps = successful_iterations / sum(timings) if sum(timings) > 0 else 0
    throughput_bps = sum(total_bytes_streamed_list) / sum(timings) if sum(timings) > 0 else 0

    return {
        "successful_iterations": successful_iterations,
        "avg_stream_time_s": avg_time,
        "std_dev_stream_time_s": std_dev_time,
        "avg_first_response_latency_s": avg_first_response_latency,
        "throughput_streams_per_s": 1 / avg_time if avg_time > 0 else 0,
        "throughput_msgs_per_s": throughput_mps * stream_length,
        "throughput_bytes_per_s": throughput_bps,
        "avg_bytes_per_stream": total_bytes_per_iteration,
    }

# --- Main Execution ---
def display_results(scenario: Dict[str, Any], grpc_results: Dict[str, Any], rest_results: Dict[str, Any]):
    print("\n--- Benchmark Scenario ---")
    print(f"  Payload Size: {scenario['payload_size']} bytes")
    print(f"  Stream Length: {scenario['stream_length']} messages")
    print(f"  Iterations: {scenario['iterations']}")

    print("\n--- gRPC Results ---")
    if "error" in grpc_results:
        print(f"  Error: {grpc_results['error']}")
    else:
        print(f"  Successful Iterations: {grpc_results['successful_iterations']}/{scenario['iterations']}")
        print(f"  Avg. Stream Time: {grpc_results['avg_stream_time_s']:.4f} s (StdDev: {grpc_results['std_dev_stream_time_s']:.4f} s)")
        print(f"  Avg. First Response Latency: {grpc_results['avg_first_response_latency_s']:.4f} s")
        print(f"  Throughput (Streams/s): {grpc_results['throughput_streams_per_s']:.2f}")
        print(f"  Throughput (Msgs/s): {grpc_results['throughput_msgs_per_s']:.2f}")
        print(f"  Throughput (Bytes/s): {grpc_results['throughput_bytes_per_s']:.2f}")
        print(f"  Avg. Bytes per Stream: {grpc_results['avg_bytes_per_stream']:.0f}")


    print("\n--- REST Results ---")
    if "error" in rest_results:
        print(f"  Error: {rest_results['error']}")
    else:
        print(f"  Successful Iterations: {rest_results['successful_iterations']}/{scenario['iterations']}")
        print(f"  Avg. Stream Time: {rest_results['avg_stream_time_s']:.4f} s (StdDev: {rest_results['std_dev_stream_time_s']:.4f} s)")
        print(f"  Avg. First Response Latency: {rest_results['avg_first_response_latency_s']:.4f} s")
        print(f"  Throughput (Streams/s): {rest_results['throughput_streams_per_s']:.2f}")
        print(f"  Throughput (Msgs/s): {rest_results['throughput_msgs_per_s']:.2f}")
        print(f"  Throughput (Bytes/s): {rest_results['throughput_bytes_per_s']:.2f}")
        print(f"  Avg. Bytes per Stream: {rest_results['avg_bytes_per_stream']:.0f}")
    print("="*40)

async def main():
    parser = argparse.ArgumentParser(description="Benchmark gRPC vs REST streaming.")
    parser.add_argument("--grpc_host", type=str, default=DEFAULT_HOST, help="gRPC server host.")
    parser.add_argument("--grpc_port", type=int, default=DEFAULT_GRPC_PORT, help="gRPC server port.")
    parser.add_argument("--rest_host", type=str, default=DEFAULT_HOST, help="REST server host.")
    parser.add_argument("--rest_port", type=int, default=DEFAULT_REST_PORT, help="REST server port.")
    parser.add_argument("--rest_endpoint", type=str, default="/stream", help="REST streaming endpoint.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Number of iterations per scenario.")
    parser.add_argument("--payload_sizes", type=str, default="10,1000,100000", help="Comma-separated payload sizes in bytes.")
    parser.add_argument("--stream_lengths", type=str, default="10,100,500", help="Comma-separated stream lengths (number of messages).")
    parser.add_argument("--skip_grpc", action="store_true", help="Skip gRPC benchmarks.")
    parser.add_argument("--skip_rest", action="store_true", help="Skip REST benchmarks.")


    args = parser.parse_args()

    payload_sizes = [int(s) for s in args.payload_sizes.split(",")]
    stream_lengths = [int(s) for s in args.stream_lengths.split(",")]

    print("Starting benchmark...")
    print(f"Iterations per scenario: {args.iterations}")
    print(f"Payload sizes: {payload_sizes}")
    print(f"Stream lengths: {stream_lengths}")
    print(f"gRPC Server: {args.grpc_host}:{args.grpc_port}")
    print(f"REST Server: {args.rest_host}:{args.rest_port}{args.rest_endpoint}")
    print("Important: Ensure both gRPC and REST servers are running and configured for streaming.")
    print("The REST server must accept POST requests at the specified endpoint, expecting a JSON payload like")
    print("{'data': 'your_payload', 'stream_length': number_of_messages} and stream back line-delimited text responses.")


    scenarios = [
        {"payload_size": ps, "stream_length": sl, "iterations": args.iterations}
        for ps in payload_sizes
        for sl in stream_lengths
    ]

    for scenario in scenarios:
        grpc_results = {"error": "Skipped"}
        rest_results = {"error": "Skipped"}

        if not args.skip_grpc:
            grpc_results = await benchmark_grpc_stream(
                args.grpc_host, args.grpc_port,
                scenario["payload_size"], scenario["stream_length"], scenario["iterations"]
            )
        
        if not args.skip_rest:
            rest_results = await benchmark_rest_stream(
                args.rest_host, args.rest_port,
                scenario["payload_size"], scenario["stream_length"], scenario["iterations"],
                endpoint=args.rest_endpoint
            )
        
        display_results(scenario, grpc_results, rest_results)

if __name__ == "__main__":
    asyncio.run(main())
