Run the iris_classifier service in production mode:

| Protocol | Command                                     |
| -------- | ------------------------------------------- |
| HTTP     | `bentoml serve-http iris_classifier:latest` |
| gRPC     | `bentoml serve-grpc iris_classifier:latest` |

Start locust testing client:

```bash
locust --class-picker -H http://localhost:3000
```
