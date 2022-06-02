from setuptools import find_packages, setup

#Read the contents of the README file
with open("README.txt", "r") as fh:
    long_description = fh.read()

setup(
    name='tegdet',
    packages=find_packages(include=['tegdet']),
    version='1.0.0',
    description='Time Evolving Graph detectors library',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/DiasporeUnizar/TEG",
    author='Diaspore team',
    author_email="simonab@unizar.es",
    license='GPL-2',
    classifiers=[
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent"
    ],
    python_requires='>=3.6.1',
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=6.2.2'],
    test_suite='test'
)