from setuptools import setup, find_packages

setup(
    name="z-adapter",
    version="0.1.0",
    author="ZNX Team",
    description="TPU-Native Bottleneck Adapter for Efficient LLM Fine-Tuning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
    ],
    extras_require={
        "tpu": ["torch_xla", "cloud-tpu-client"],
        "dev": ["pytest>=7.0"],
    },
)
