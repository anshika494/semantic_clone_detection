"""
Setup configuration for Semantic Code Clone Detection System.
"""

from setuptools import setup, find_packages

setup(
    name="semantic-clone-detection",
    version="1.0.0",
    description="Research-grade semantic code clone detection using ASTs and CodeBERT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # Core ML
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",

        # AST Parsing
        "tree-sitter>=0.20.1",
        "javalang>=0.13.0",

        # Scientific computing
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",

        # Data handling
        "pandas>=2.0.0",
        "sqlite3",  # stdlib

        # Configuration & utilities
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "click>=8.1.0",

        # Visualization & metrics
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",

        # Caching & serialization
        "joblib>=1.2.0",
        "h5py>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "scd-train=scripts.train:main",
            "scd-infer=scripts.inference:main",
            "scd-eval=scripts.evaluate:main",
            "scd-preprocess=scripts.preprocess:main",
        ]
    },
)
