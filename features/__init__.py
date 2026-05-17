"""Feature engineering package for the quant research platform."""

from features.microstructure import (
    VWAPBands,
    aggregate_order_book_imbalance,
    anchored_vwap,
    build_microstructure_features,
    classify_trade_direction,
    cumulative_delta,
    delta_divergence,
    liquidity_sweep_detection,
    order_book_imbalance,
    order_flow_pressure_signal,
    volume_delta,
    vwap_deviation,
    vwap_standard_deviation_bands,
)
from features.regimes import RegimeDetector, RegimeDetectorConfig
from features.targets import TargetBuilder, build_supervised_frame
from features.technical import FeatureEngineer, build_feature_matrix

__all__ = [
    "FeatureEngineer",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "TargetBuilder",
    "VWAPBands",
    "aggregate_order_book_imbalance",
    "anchored_vwap",
    "build_feature_matrix",
    "build_microstructure_features",
    "build_supervised_frame",
    "classify_trade_direction",
    "cumulative_delta",
    "delta_divergence",
    "liquidity_sweep_detection",
    "order_book_imbalance",
    "order_flow_pressure_signal",
    "volume_delta",
    "vwap_deviation",
    "vwap_standard_deviation_bands",
]
