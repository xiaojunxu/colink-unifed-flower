import sys

from unifed.frameworks.flower import protocol
from unifed.frameworks.flower.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here

