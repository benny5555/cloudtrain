"""Configuration management for CloudTrain.

This module provides configuration and credential management functionality
for the CloudTrain universal cloud training API.
"""

from cloudtrain.config.manager import ConfigManager
from cloudtrain.config.settings import CloudTrainSettings

__all__ = ["ConfigManager", "CloudTrainSettings"]
