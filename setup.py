"""Setup configuration for SelfLLM package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="selfllm",
    version="0.1.0",
    description="A recursively self-improving foundation language model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SelfLLM Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    package_data={
        "selfllm": ["config.yaml"],
    },
    entry_points={
        "console_scripts": [
            "selfllm=selfllm.train:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm transformer self-improving machine-learning",
    project_urls={
        "Source": "https://github.com/example/selfllm",
    },
)
