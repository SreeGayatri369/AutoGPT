from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModelWithSummary(BaseModel, ABC):
    
    def summary(self) -> str:
            return str(self)



