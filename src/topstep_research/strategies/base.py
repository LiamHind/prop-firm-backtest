from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topstep_research.config import StrategyConfig


@dataclass
class StrategyContext:
    strategy_name: str
    strategy_config: StrategyConfig
    extra_params: dict[str, object]
    random_seed: int


class BaseStrategy(ABC):
    def __init__(self, context: StrategyContext) -> None:
        self.context = context
        self.strategy_config = context.strategy_config

    @property
    def name(self) -> str:
        return self.context.strategy_name

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def zero_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
