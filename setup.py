from setuptools import find_packages, setup

setup(
    name='tegdet',
    packages=find_packages(include=['tegdet']),
    version='1.0',
    description='Time Evolving Graph detectors library',
    author='Diaspore team',
    license='MIT',
    install_requires=[
        'numpy>=1.20.1',
        'pandas>=1.2.3',
        'scipy>=1.6.2'
    ],
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=6.2.2'],
    test_suite='test',
)