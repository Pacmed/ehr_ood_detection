from setuptools import find_packages, setup

setup(
    name='uncertainty_estimation',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.1.2',
        'numpy==1.16.0',
        'pandas==0.25.0',
        'torch==1.3.1',
        'scikit-learn==0.22',
        'seaborn==0.9.0'
    ]
)