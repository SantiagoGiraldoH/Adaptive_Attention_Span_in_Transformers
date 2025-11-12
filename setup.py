from setuptools import setup, find_packages

setup(
    name="adaptive-attention-span",
    version="0.1.0",
    author="Tu Nombre",
    description="Implementation of Adaptive Attention Span in Transformers",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
    ],
)
