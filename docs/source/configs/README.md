# BentoML Prometheus and Grafana Example Stack

Users can freely update [prometheus.yml](./prometheus/prometheus.yml) target section to define what should be  monitored by Prometheus.

`./grafana/provisioning` provides both `datasources` and `dashboards` for us to specify datasources and bootstrap our dashboards quickly thanks to the [Introduction of Provisioning in v5.0.0](https://grafana.com/docs/grafana/latest/administration/provisioning/)

If you would like to automate the installation of additional dashboards just copy the Dashboard JSON file to `/grafana/provisioning/dashboards` and it will be provisioned next time you stop and start Grafana.

Please refers to [BentoML's guides on Monitoring with Prometheus](https://docs.bentoml.org/en/latest/guides/monitoring.html#docker-compose-stack-for-prometheus-and-grafana)