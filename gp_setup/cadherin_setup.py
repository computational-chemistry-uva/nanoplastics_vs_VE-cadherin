import numpy as np
from buq.systems import CollectiveVariableSystem

import numpy as np
from buq.systems import CollectiveVariableSystem
from buq.systems import CollectiveVariableSystem

class Cadherin(CollectiveVariableSystem):
    def __init__(self, bounds):
        super().__init__(dim=2, bounds=bounds)

    def write_plumed_input(self, x):
        pass

    def run_simulation(self, x):
        pass
    def get_force(self, x):
        pass