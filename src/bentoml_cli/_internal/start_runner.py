import sys

from .start import start_command

if __name__ == "__main__":
    start_command.main(["start-runner-server", *sys.argv[1:]])
