"""
Setup script for MTS Data Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mts-data-pipeline",
    version="1.0.0",
    author="MTS Project Team",
    author_email="mts@example.com",
    description="Complete data preparation pipeline for MTS Music Generation Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mts-project/data-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "audio": [
            "pyrubberband>=0.3.0",
            "pedalboard>=0.7.0",
        ],
        "gpu": [
            "torch-audio>=0.12.0",
            "torchaudio>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mts-pipeline=pipeline:main",
        ],
    },
)
