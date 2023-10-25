#!/usr/bin/env python

import pdb
import sys
import os
import json


class Simulation:
    """
    Base class to handle parameters
    """

    def __init__(self, f_params=None):
        # mesh parameters
        self.p = {}

        # set mesh parameters (load from json file if given)
        if f_params is None:
            self.set_params()
        else:
            self.load_params(f_params)

        # set default parameters
        self.set_defaults()

        # validate parameters
        self.validate_params()

    def set_params(self):
        """
        Manually set parameters (e.g. to default values)
        """
        raise ValueError("Implement set_params in derived class")

    def set_defaults(self):
        """
        Validate parameters
        """
        # raise ValueError("Implement set_defaults in derived class")
        pass

    def validate_params(self):
        """
        Validate parameters
        """
        # raise ValueError("Implement validate_params in derived class")
        pass

    def load_params(self, file_name):
        """
        Load parameters from json file_name
        """
        # read parameters from json file
        with open(file_name, "r") as file:
            param = json.load(file)

        # set parameters
        for k, v in param.items():
            self.p[k] = v

    def save_params(self, file_name):
        """
        Save parameters to json file_name
        """
        # save parameters to json file
        file_name = os.path.join(self.p["f_out"], file_name)
        with open(file_name, "w") as file:
            json.dump(self.p, file, indent=4, sort_keys=True)
