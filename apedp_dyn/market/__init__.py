"""APEDP Real Market Sweeps: rolling windows, regimes, and multi-asset discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apedp_dyn._lazy import lazy_exports

if TYPE_CHECKING:  # the names below, for type checkers and readers; nothing is imported at runtime
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

# Public name -> the module that defines it. Resolved on first access, so
# importing this package (which importing any of its submodules does) costs
# nothing and pulls in nothing. See apedp_dyn/_lazy.py.
_EXPORTS = {
    "AVAILABLE_ASSETS": "apedp_dyn.market.assets",
    "compute_strictly_causal_features": "apedp_dyn.market.assets",
    "load_series_with_dates": "apedp_dyn.market.assets",
    "LobTransportRecovery": "apedp_dyn.market.lob_transport",
    "SpectralTransportRecovery": "apedp_dyn.market.lob_transport",
    "discover_transport_with_engine": "apedp_dyn.market.lob_transport",
    "recover_transport_coefficients": "apedp_dyn.market.lob_transport",
    "simulate_lob_transport": "apedp_dyn.market.lob_transport",
    "snapshot_centre_of_mass": "apedp_dyn.market.lob_transport",
    "snapshot_mass": "apedp_dyn.market.lob_transport",
    "snapshot_variance": "apedp_dyn.market.lob_transport",
    "spectral_transport_recovery": "apedp_dyn.market.lob_transport",
    "MarketRegime": "apedp_dyn.market.regimes",
    "REGIME_METADATA": "apedp_dyn.market.regimes",
    "classify_market_regime": "apedp_dyn.market.regimes",
    "MarketSweepSummary": "apedp_dyn.market.sweep",
    "WindowDiscoveryRecord": "apedp_dyn.market.sweep",
    "execute_window_discovery": "apedp_dyn.market.sweep",
    "run_market_sweep": "apedp_dyn.market.sweep",
    "WINDOWS": "apedp_dyn.market.windows",
    "WindowSpec": "apedp_dyn.market.windows",
    "extensive_rolling_windows": "apedp_dyn.market.windows",
    "generated_rolling_windows": "apedp_dyn.market.windows",
    "get_window_by_id": "apedp_dyn.market.windows",
}

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

__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
