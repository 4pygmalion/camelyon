from setuptools import find_packages, setup

setup(
    name="camelyon",
    version="0.4",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=["openslide-python", "matplotlib", "numpy"],
)