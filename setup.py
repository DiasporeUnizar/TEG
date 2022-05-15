from setuptools import find_packages, setup

setup(
    name='tegdet',
    packages=find_packages(include=['tegdet']),
    version='2.0.0',
    description='Time Evolving Graph detectors library',
    author='Diaspore team',
    license='MIT',
    install_requires=[
        'numpy>=1.14.5',
        'pandas>=1.1',
        'scipy>=1.5'
    ],
    python_requires='>=3.6.1',
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=6.2.2'],
    test_suite='test',
)