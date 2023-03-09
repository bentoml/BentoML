import bentoml

deploy = bentoml.ray.deployment(
    "fraud_detection:latest",
    {"num_replicas": 5, "ray_actor_options": {"num_cpus": 1}},
    {
        "ieee-fraud-detection-sm": {
            "num_replicas": 1,
            "ray_actor_options": {"num_cpus": 5},
        }
    },
    enable_batching=True,
    batching_config={
        "ieee-fraud-detection-sm": {
            "predict_proba": {"max_batch_size": 5, "batch_wait_timeout_s": 0.2}
        }
    },
)
