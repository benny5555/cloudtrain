"""Mock provider implementation for testing CloudTrain.

This module provides a mock cloud provider implementation that can be used
for testing and development without requiring actual cloud provider credentials.
"""

from cloudtrain.providers.mock.provider import MockProvider

__all__ = ["MockProvider"]
