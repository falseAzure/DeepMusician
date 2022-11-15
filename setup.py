from setuptools import find_packages, setup

setup(
    name="DeepMusician",
    version="0.0.1",
    description="DeepMusician: A Deep Learning Approach to Music Generation",
    author=["falseAzure"],
    author_email=[
        "falseazure@proton.me",
    ],
    license="MIT",
    install_requires=[
        "numpy==1.23.3",
        "scipy==1.9.3",
        "pandas==1.5.1",
        "scikit-learn==1.1.3",
        "torch==1.13",
        "ipykernel==6.17.0",
        "jupyter==1.0.0",
        "black==22.10.0",
        "flake8==5.0.4",
        "pytest==7.2.0",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)