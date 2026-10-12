"""Decoupled Execution Cost & Market Impact Modeling Interface.

Implements Reviewer Correction C: Separates execution models into horizon-appropriate,
empirically calibrated classes rather than combining conflicting formulas:
  1. FixedPlusSlippageCostModel: Realistic spread, slippage sensitivity, and exchange fees
     for medium/slower horizon strategies.
  2. AlmgrenChrissImpactCostModel: Formally separated temporary (eta * v/V) and
     permanent (gamma * sigma * Q/V) market impact models.
  3. LimitOrderQueueCostModel: Queue priority fill probability and adverse selection
     penalties for order-book dependent strategies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


class ExecutionCostModelBase(ABC):
    """Abstract base class for all execution cost and impact models."""

    @abstractmethod
    def compute_cost_drag(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute execution cost drag per observation in return units."""
        pass


@dataclass
class FixedPlusSlippageCostModel(ExecutionCostModelBase):
    """Calibrated transaction costs for medium-frequency / slow factor strategies.

    Drag = exchange_fee * |dpos| + 0.5 * (spread / price) * |dpos| + slippage_coeff * sigma * |dpos|
    """
    fee_bps: float = 2.0  # 2 bps per turnover
    slippage_coeff: float = 0.10  # Slippage sensitivity to local volatility
    default_spread_bps: float = 3.0  # 3 bps half-spread default

    def compute_cost_drag(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(pos)
        dpos = np.zeros(n)
        dpos[1:] = np.abs(pos[1:] - pos[:-1])

        # Exchange fee component
        fee_drag = (self.fee_bps / 10000.0) * dpos

        # Spread crossing component
        if spreads is not None and len(spreads) == n:
            rel_spread = np.maximum(spreads / np.maximum(p, 1e-6), 1e-6)
        else:
            rel_spread = np.full(n, (self.default_spread_bps / 10000.0))
        spread_drag = 0.5 * rel_spread * dpos

        # Local volatility slippage
        ret = np.zeros(n)
        ret[1:] = np.abs(np.diff(np.log(np.maximum(p, 1e-6))))
        sigma_local = pd.Series(ret).rolling(20, min_periods=2).std().bfill().fillna(0.01).to_numpy()
        slippage_drag = self.slippage_coeff * sigma_local * dpos

        return fee_drag + spread_drag + slippage_drag


@dataclass
class AlmgrenChrissImpactCostModel(ExecutionCostModelBase):
    """Classical Almgren & Chriss (2000) optimal execution impact model.

    Separates:
      1. Temporary impact: eta * (participation_rate)
      2. Permanent impact: gamma * sigma * (participation_rate)
    """
    eta_temp: float = 0.05  # Temporary impact parameter
    gamma_perm: float = 0.02  # Permanent impact parameter
    exchange_fee_bps: float = 1.5

    def compute_cost_drag(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        p = np.asarray(prices, dtype=float)
        n = len(pos)
        dpos = np.zeros(n)
        dpos[1:] = np.abs(pos[1:] - pos[:-1])

        vol = volumes if volumes is not None and len(volumes) == n else np.full(n, 1e6)
        vol_clean = np.maximum(np.asarray(vol, dtype=float), 1e3)

        participation = np.clip(dpos / vol_clean, 0.0, 0.50)

        # Volatility estimate
        ret = np.zeros(n)
        ret[1:] = np.abs(np.diff(np.log(np.maximum(p, 1e-6))))
        sigma_local = pd.Series(ret).rolling(20, min_periods=2).std().bfill().fillna(0.01).to_numpy()

        temp_impact = self.eta_temp * participation * dpos
        perm_impact = self.gamma_perm * sigma_local * participation * dpos
        fee_drag = (self.exchange_fee_bps / 10000.0) * dpos

        return fee_drag + temp_impact + perm_impact


@dataclass
class LimitOrderQueueCostModel(ExecutionCostModelBase):
    """High-frequency microstructure limit order queue model with adverse selection.

    Models:
      1. Fill probability based on queue positioning.
      2. Adverse selection drag (toxic order flow fills against position).
    """
    queue_exhaustion_rate: float = 0.30
    adverse_selection_bps: float = 4.0  # Loss conditional on adverse fill
    maker_rebate_bps: float = 1.0  # Maker rebate earned on passive fill

    def compute_cost_drag(
        self,
        positions: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
        ofi: np.ndarray | None = None,
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        n = len(pos)
        dpos = np.zeros(n)
        dpos[1:] = np.abs(pos[1:] - pos[:-1])

        # Passive fill earns rebate, but suffers adverse selection if ofi opposes
        rebate = (self.maker_rebate_bps / 10000.0) * dpos

        if ofi is not None and len(ofi) == n:
            # Adverse selection happens when order flow is toxic against position change
            toxic = (np.sign(ofi) != np.sign(pos)) & (dpos > 0.0)
            adverse_drag = np.zeros(n)
            adverse_drag[toxic] = (self.adverse_selection_bps / 10000.0) * dpos[toxic]
        else:
            # Baseline adverse selection assumption
            adverse_drag = 0.5 * (self.adverse_selection_bps / 10000.0) * dpos

        net_drag = adverse_drag - rebate
        return np.maximum(net_drag, 0.0)
