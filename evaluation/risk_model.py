"""Factor risk modeling and return attribution utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRegressionResult:
    """OLS factor regression result for strategy risk analysis."""

    alpha: float
    betas: pd.Series
    r_squared: float
    fitted_returns: pd.Series
    residual_returns: pd.Series


@dataclass(frozen=True)
class ReturnAttribution:
    """Factor and idiosyncratic contribution summary."""

    factor_contributions: pd.Series
    idiosyncratic_contribution: float
    total_return: float
    attribution_table: pd.DataFrame


def align_strategy_and_factors(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Align strategy returns and factor returns on shared timestamps."""
    frame = factor_returns.join(strategy_returns.rename("strategy_return"), how="inner").dropna()
    if frame.empty:
        raise ValueError("No overlapping non-null strategy and factor returns.")
    return frame["strategy_return"], frame[factor_returns.columns]


def ols_factor_regression(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    include_intercept: bool = True,
) -> FactorRegressionResult:
    """Regress strategy returns on one or more factor return series using OLS."""
    aligned_strategy, aligned_factors = align_strategy_and_factors(strategy_returns, factor_returns)
    x = aligned_factors.to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x]) if include_intercept else x
    y = aligned_strategy.to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x_design, y, rcond=None)

    alpha = float(coefficients[0]) if include_intercept else 0.0
    beta_values = coefficients[1:] if include_intercept else coefficients
    fitted = pd.Series(x_design @ coefficients, index=aligned_strategy.index, name="fitted_return")
    residuals = pd.Series(y - fitted.to_numpy(), index=aligned_strategy.index, name="residual_return")
    total_sum_squares = float(((aligned_strategy - aligned_strategy.mean()) ** 2).sum())
    residual_sum_squares = float((residuals**2).sum())
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 0.0

    return FactorRegressionResult(
        alpha=alpha,
        betas=pd.Series(beta_values, index=aligned_factors.columns, name="beta"),
        r_squared=float(r_squared),
        fitted_returns=fitted,
        residual_returns=residuals,
    )


def beta_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Return strategy beta to a benchmark such as SPY."""
    result = ols_factor_regression(strategy_returns, benchmark_returns.to_frame("benchmark"))
    return float(result.betas["benchmark"])


def jensens_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    annualization_factor: int = 252,
) -> float:
    """Return annualized Jensen's alpha after benchmark beta adjustment."""
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if isinstance(risk_free_rate, pd.Series):
        aligned = aligned.join(risk_free_rate.rename("risk_free"), how="inner").dropna()
        risk_free = aligned["risk_free"]
    else:
        risk_free = pd.Series(risk_free_rate / annualization_factor, index=aligned.index)

    excess_strategy = aligned["strategy"] - risk_free
    excess_benchmark = aligned["benchmark"] - risk_free
    result = ols_factor_regression(excess_strategy, excess_benchmark.to_frame("benchmark"))
    return float(result.alpha * annualization_factor)


def factor_exposure_correlations(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> pd.Series:
    """Return strategy correlation with each supplied factor series."""
    aligned_strategy, aligned_factors = align_strategy_and_factors(strategy_returns, factor_returns)
    return aligned_factors.corrwith(aligned_strategy).rename("correlation")


def fama_french_exposures(
    strategy_returns: pd.Series,
    fama_french_factors: pd.DataFrame,
    factor_columns: tuple[str, ...] = ("MKT", "SMB", "HML", "UMD"),
) -> pd.DataFrame:
    """Return regression betas and correlations for Fama-French-style factors."""
    available_columns = [column for column in factor_columns if column in fama_french_factors]
    if not available_columns:
        raise KeyError(f"None of the requested factor columns are available: {factor_columns}")

    factors = fama_french_factors[available_columns]
    regression = ols_factor_regression(strategy_returns, factors)
    correlations = factor_exposure_correlations(strategy_returns, factors)
    return pd.DataFrame({"beta": regression.betas, "correlation": correlations})


def volatility_factor_correlation(
    strategy_returns: pd.Series,
    vix: pd.Series,
    use_changes: bool = True,
) -> float:
    """Return strategy correlation with daily VIX changes or levels."""
    volatility_factor = vix.diff() if use_changes else vix
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), volatility_factor.rename("vix_factor")],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned["strategy"].corr(aligned["vix_factor"]))


def return_attribution(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> ReturnAttribution:
    """Attribute total strategy return to factor and idiosyncratic components."""
    regression = ols_factor_regression(strategy_returns, factor_returns)
    aligned_strategy, aligned_factors = align_strategy_and_factors(strategy_returns, factor_returns)
    factor_period_contributions = aligned_factors.multiply(regression.betas, axis=1)
    factor_contributions = factor_period_contributions.sum()
    alpha_contribution = regression.alpha * len(aligned_strategy)
    residual_contribution = float(regression.residual_returns.sum())
    idiosyncratic = float(alpha_contribution + residual_contribution)
    total_return = float(aligned_strategy.sum())

    attribution_table = pd.DataFrame(
        {
            "contribution": pd.concat(
                [factor_contributions, pd.Series({"alpha_plus_residual": idiosyncratic})],
            ),
        },
    )
    attribution_table["share_of_total"] = (
        attribution_table["contribution"] / total_return if total_return else np.nan
    )
    return ReturnAttribution(
        factor_contributions=factor_contributions,
        idiosyncratic_contribution=idiosyncratic,
        total_return=total_return,
        attribution_table=attribution_table,
    )


def covariance_matrix(
    returns: pd.DataFrame,
    annualization_factor: int | None = 252,
) -> pd.DataFrame:
    """Return covariance matrix for multi-asset portfolio risk."""
    covariance = returns.dropna(how="all").cov()
    return covariance * annualization_factor if annualization_factor else covariance


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Return portfolio volatility from weights and covariance matrix."""
    aligned_weights = weights.reindex(covariance.index).fillna(0.0)
    variance = float(aligned_weights.to_numpy() @ covariance.to_numpy() @ aligned_weights.to_numpy())
    return float(np.sqrt(max(variance, 0.0)))
