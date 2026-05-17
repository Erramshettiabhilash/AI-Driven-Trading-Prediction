"""Model package for supervised financial machine learning."""

from models.factor_dataset import FactorDataset, build_factor_dataset
from models.ensemble import (
    EnsembleEvaluation,
    LinearRegressionBaseline,
    RidgeStackingEnsemble,
    evaluate_prediction_signals,
    ic_weighted_ensemble,
    regime_conditional_ensemble,
    regime_conditional_weights,
    rolling_model_ic,
)
from models.lstm_timeseries import (
    LSTMEvaluation,
    LSTMModelConfig,
    LSTMTimeSeriesModel,
    SequenceDataset,
    create_lstm_sequences,
    subset_sequence_dataset,
    time_series_sequence_splits,
)
from models.regime_aware import RegimeAwareXGBoostModel
from models.xgboost_factor import XGBoostEvaluation, XGBoostFactorModel, XGBoostModelConfig

__all__ = [
    "FactorDataset",
    "EnsembleEvaluation",
    "LSTMEvaluation",
    "LinearRegressionBaseline",
    "LSTMModelConfig",
    "LSTMTimeSeriesModel",
    "RidgeStackingEnsemble",
    "RegimeAwareXGBoostModel",
    "SequenceDataset",
    "XGBoostEvaluation",
    "XGBoostFactorModel",
    "XGBoostModelConfig",
    "build_factor_dataset",
    "create_lstm_sequences",
    "evaluate_prediction_signals",
    "ic_weighted_ensemble",
    "regime_conditional_ensemble",
    "regime_conditional_weights",
    "rolling_model_ic",
    "subset_sequence_dataset",
    "time_series_sequence_splits",
]
