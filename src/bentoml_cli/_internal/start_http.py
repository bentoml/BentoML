import sys

from .start import start_command

if __name__ == "__main__":
    start_command.main(["start-http-server", *sys.argv[1:]])
