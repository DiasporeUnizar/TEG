from setuptools import find_packages, setup

setup(
    name='tegdet',
    packages=find_packages(include=['tegdet']),
    version='2.0.0',
    description='Time Evolving Graph detectors library',
    author='Diaspore team',
    license='MIT',
    install_requires=[
        'numpy>=1.22',
        'pandas>=1.4',
        'scipy>=1.8'
    ],
    python_requires='>=3.10',
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=6.2.2'],
    test_suite='test',
)