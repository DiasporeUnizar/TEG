from setuptools import find_packages, setup

#Read the contents of the README file
#from pathlib import Path
#this_directory = Path(__file__).parent
#long_description = (this_directory / "README.md").read_text()

setup(
    name='tegdet',
    packages=find_packages(include=['tegdet']),
    version='2.0.0',
    description='Time Evolving Graph detectors library',
    long_description="The library includes the implementation of Time-Evolving-Graph detectors.",
    long_description_content_type="text/markdown",
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
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=1.1',
        'scipy>=1.5'
    ],
    python_requires='>=3.6.1',
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=6.2.2'],
    test_suite='test'
)