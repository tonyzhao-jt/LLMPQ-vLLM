from pathlib import Path

from setuptools import find_packages, setup


def get_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, "r", encoding="utf-8") as file:
        requirements = file.read().splitlines()
    requirements = [
        req.strip()
        for req in requirements
        if req.strip() and not req.startswith("#")  # noqa
    ]
    return requirements


setup(
    name="llmpq",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={
        'console_scripts': [
            'llmpq-algo = llmpq.optimizer.algo.entry:algo_main',
        ]
    },
    author="Tony",
    author_email="juntaozh@connect.hku.hk",
    description="Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization",  # noqa
    url="https://github.com/tonyzhao-jt/LLM-PQ/tree/main",
)
