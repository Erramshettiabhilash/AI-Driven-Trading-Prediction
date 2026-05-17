from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    """Verify the Step 1 quant platform directory scaffold exists."""
    required_directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "data/live",
        "features",
        "models",
        "evaluation",
        "optimization",
        "rl",
        "explainability",
        "live",
        "visualization",
        "notebooks",
        "tests",
        "results",
        "config",
    ]

    for directory in required_directories:
        assert (PROJECT_ROOT / directory).is_dir(), f"Missing directory: {directory}"


def test_config_file_loads() -> None:
    """Verify the YAML config is readable and contains core project sections."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = [
        "project",
        "paths",
        "data",
        "preprocessing",
        "features",
        "targets",
        "research",
        "models",
        "evaluation",
        "risk",
        "live",
        "mlflow",
    ]

    for section in required_sections:
        assert section in config, f"Missing config section: {section}"


def test_importable_research_packages_exist() -> None:
    """Verify core source directories can behave as importable Python packages."""
    packages = [
        "features",
        "models",
        "evaluation",
        "optimization",
        "rl",
        "explainability",
        "live",
        "visualization",
    ]

    for package in packages:
        assert (PROJECT_ROOT / package / "__init__.py").is_file()
