def main():
    from ...utils.ngrok import start_ngrok
    from ...configuration.containers import BentoMLContainer

    start_ngrok(BentoMLContainer.api_server_config.port.get())


if __name__ == "__main__":
    main()
