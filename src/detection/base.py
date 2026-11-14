from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import PersonDetection


class PersonDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Return all people detected in a BGR frame."""
        raise NotImplementedError
