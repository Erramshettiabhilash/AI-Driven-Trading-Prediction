import numpy as np
import pandas as pd

from evaluation import (
    beta_to_benchmark,
    covariance_matrix,
    fama_french_exposures,
    jensens_alpha,
    ols_factor_regression,
    portfolio_volatility,
    return_attribution,
    volatility_factor_correlation,
)


def sample_index() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=100, freq="D", tz="UTC", name="timestamp")


def test_ols_factor_regression_recovers_known_alpha_and_beta() -> None:
    index = sample_index()
    benchmark = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index, name="SPY")
    strategy = 0.0004 + 1.5 * benchmark

    result = ols_factor_regression(strategy, benchmark.to_frame())

    assert np.isclose(result.alpha, 0.0004)
    assert np.isclose(result.betas["SPY"], 1.5)
    assert result.r_squared > 0.99


def test_beta_to_benchmark_and_jensens_alpha() -> None:
    index = sample_index()
    benchmark = pd.Series(np.sin(np.arange(len(index)) / 8) * 0.01, index=index)
    strategy = 0.0002 + 0.8 * benchmark

    beta = beta_to_benchmark(strategy, benchmark)
    alpha = jensens_alpha(strategy, benchmark, annualization_factor=252)

    assert np.isclose(beta, 0.8)
    assert np.isclose(alpha, 0.0002 * 252)


def test_fama_french_exposures_returns_betas_and_correlations() -> None:
    index = sample_index()
    rng = np.random.default_rng(42)
    factors = pd.DataFrame(
        {
            "MKT": rng.normal(0.0, 0.01, len(index)),
            "SMB": rng.normal(0.0, 0.006, len(index)),
            "HML": rng.normal(0.0, 0.004, len(index)),
            "UMD": rng.normal(0.0, 0.008, len(index)),
        },
        index=index,
    )
    strategy = 0.0001 + factors @ pd.Series({"MKT": 1.2, "SMB": -0.4, "HML": 0.3, "UMD": 0.0})

    exposures = fama_french_exposures(strategy, factors)

    assert {"beta", "correlation"}.issubset(exposures.columns)
    assert np.isclose(exposures.loc["MKT", "beta"], 1.2, atol=1e-10)
    assert np.isclose(exposures.loc["SMB", "beta"], -0.4, atol=1e-10)


def test_volatility_factor_correlation_uses_vix_changes() -> None:
    index = sample_index()
    vix = pd.Series(np.linspace(18.0, 28.0, len(index)), index=index)
    strategy = vix.diff().fillna(0.0) * -0.01

    correlation = volatility_factor_correlation(strategy, vix, use_changes=True)

    assert correlation < -0.99


def test_return_attribution_sums_to_total_return() -> None:
    index = sample_index()
    factors = pd.DataFrame(
        {
            "MKT": np.sin(np.arange(len(index)) / 5) * 0.01,
            "SMB": np.cos(np.arange(len(index)) / 6) * 0.006,
        },
        index=index,
    )
    strategy = 0.0003 + factors["MKT"] * 0.7 + factors["SMB"] * -0.2

    attribution = return_attribution(strategy, factors)

    reconstructed = attribution.factor_contributions.sum() + attribution.idiosyncratic_contribution
    assert np.isclose(reconstructed, attribution.total_return)
    assert "alpha_plus_residual" in attribution.attribution_table.index


def test_covariance_matrix_and_portfolio_volatility() -> None:
    index = sample_index()
    returns = pd.DataFrame(
        {
            "asset_a": np.sin(np.arange(len(index)) / 4) * 0.01,
            "asset_b": np.cos(np.arange(len(index)) / 6) * 0.008,
        },
        index=index,
    )
    covariance = covariance_matrix(returns, annualization_factor=252)
    weights = pd.Series({"asset_a": 0.6, "asset_b": 0.4})

    volatility = portfolio_volatility(weights, covariance)

    assert covariance.shape == (2, 2)
    assert volatility > 0.0
