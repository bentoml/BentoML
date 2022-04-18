def main():
    from ...utils.ngrok import start_ngrok
    from ...configuration.containers import DeploymentContainer

    start_ngrok(DeploymentContainer.api_server_config.port.get())


if __name__ == "__main__":
    main()
