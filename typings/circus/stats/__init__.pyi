"""
This type stub file was generated by pyright.
"""

import sys
import signal
import argparse
from circus.stats.streamer import StatsStreamer
from circus.util import configure_logger
from circus.sighandler import SysHandler
from circus import __version__, logger, util

"""
Stats architecture:

 * streamer.StatsStreamer listens to circusd events and maintain a list of pids
 * collector.StatsCollector runs a pool of threads that compute stats for each
   pid in the list. Each stat is pushed in a queue
 * publisher.StatsPublisher continuously pushes those stats in a zmq PUB socket
 * client.StatsClient is a simple subscriber that can be used to intercept the
   stream of stats.
"""
def main():
    ...

if __name__ == '__main__':
    ...
