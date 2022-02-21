import os, dill
from prima.configuration import Experiment

name = "prima"
version = "0.4.28"
experiment = Experiment()
configs = experiment.configs

if os.name == "nt":
    dill.settings["recurse"] = True

logo = r"""
                  ___________    ____
  ________ __ ___/ / ____/   |  / __ \
 / ___/ __` / __  / /   / /| | / / / /
/ /__/ /_/ / /_/ / /___/ ___ |/ /_/ /
\___/\__,_/\__,_/\____/_/  |_/_____/
by prima
"""


from .framework import create_foundation, ParitalState

init = create_foundation
