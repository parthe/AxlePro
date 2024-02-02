from setuptools import setup, find_packages
import axlepro

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='axlepro',
    version=axlepro.__version__,
    author='Yiming Zhang, Parthe Pandit',
    author_email='yiz134@ucsd.edu, parthe1292@gmail.com',
    description='Fast solver for Kernel Regression',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/parthe/AxlePro',
    project_urls = {
        "Bug Tracker": "https://github.com/parthe/AxlePro/issues"
    },
    license='MIT license',
    packages=find_packages(),
    install_requires=requirements,
)
