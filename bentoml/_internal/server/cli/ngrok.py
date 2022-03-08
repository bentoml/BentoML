def main():
    from ...utils.ngrok import _start_ngrok
    from ...configuration.containers import DeploymentContainer

    _start_ngrok(DeploymentContainer.api_server_config.port.get())


if __name__ == "__main__":
    main()
