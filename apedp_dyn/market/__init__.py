"""APEDP Real Market Sweeps: rolling windows, regimes, and multi-asset discovery."""
from __future__ import annotations

from apedp_dyn.market.assets import (
    AVAILABLE_ASSETS,
    compute_strictly_causal_features,
    load_series_with_dates,
)
from apedp_dyn.market.lob_transport import (
    LobTransportRecovery,
    SpectralTransportRecovery,
    discover_transport_with_engine,
    recover_transport_coefficients,
    simulate_lob_transport,
    snapshot_centre_of_mass,
    snapshot_mass,
    snapshot_variance,
    spectral_transport_recovery,
)
from apedp_dyn.market.regimes import (
    REGIME_METADATA,
    MarketRegime,
    classify_market_regime,
)
from apedp_dyn.market.sweep import (
    MarketSweepSummary,
    WindowDiscoveryRecord,
    execute_window_discovery,
    run_market_sweep,
)
from apedp_dyn.market.windows import (
    WINDOWS,
    WindowSpec,
    extensive_rolling_windows,
    generated_rolling_windows,
    get_window_by_id,
)

__all__ = [
    "AVAILABLE_ASSETS",
    "REGIME_METADATA",
    "WINDOWS",
    "LobTransportRecovery",
    "MarketRegime",
    "MarketSweepSummary",
    "SpectralTransportRecovery",
    "WindowDiscoveryRecord",
    "WindowSpec",
    "classify_market_regime",
    "compute_strictly_causal_features",
    "discover_transport_with_engine",
    "execute_window_discovery",
    "extensive_rolling_windows",
    "generated_rolling_windows",
    "get_window_by_id",
    "load_series_with_dates",
    "recover_transport_coefficients",
    "run_market_sweep",
    "simulate_lob_transport",
    "snapshot_centre_of_mass",
    "snapshot_mass",
    "snapshot_variance",
    "spectral_transport_recovery",
]
