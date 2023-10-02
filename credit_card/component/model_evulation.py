import os,sys
from credit_card.logger import logging
from credit_card.exception import CreditCardException
import numpy as np
import pandas as pd
from credit_card.constant import *
from credit_card.entity.artifact_entity import DataTransformArtifact,DataValidationArtifact,ModelArtifactConfig,ModelEvulationArtifact
from credit_card.entity.config_entity import ModelEvulationConfig
from credit_card.util.util import read_yaml

class ModelEvulation:

    def __init__(self) -> None:
        pass

    