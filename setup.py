"""
Setup script for Multi-Model Financial Forecasting.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="multi-model-financial-forecasting",
    version="1.0.0",
    description="AI-assisted multi-model financial forecasting platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/multi-model-financial-forecasting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "polars>=0.20.0",
        "yfinance>=0.2.28",
        "pandas-datareader>=0.10.0",
        "TA-Lib>=0.4.28",
        "stockstats>=0.5.4",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "lightning>=2.2.0",
        "mlflow>=2.14.0",
        "huggingface_hub>=0.33.1",
        "einops>=0.8.1",
        "openpyxl>=3.1.2",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crnn-preprocess=scripts.preprocess_data:main",
            "crnn-train=scripts.train:main",
            "crnn-test=scripts.test:main",
            "crnn-validate=scripts.validate:main",
            "crnn-backtest=scripts.backtest:main",
            "crnn-predict=scripts.predict:main",
            "crnn-run=scripts.run_all:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
