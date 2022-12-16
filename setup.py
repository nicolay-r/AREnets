from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filenames):
    r_total = []
    for filename in filenames:
        with open(filename) as f:
            r_local = f.read().splitlines()
            r_total.extend(r_local)
    return r_total


setup(
    name='arenets',
    version='0.23.0',
    description='Tensorflow-based framework which lists implementation of conventional neural '
                'network models (CNN, RNN-based) for Relation Extraction classification tasks '
                'as well as API for custom model implementation',
    url='https://github.com/nicolay-r/AREnets',
    author='Nicolay Rusnachenko',
    author_email='???',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='natural language processing, relation extraction, sentiment analysis',
    packages=find_packages(),
    install_requires=get_requirements([
        'dependencies.txt'
    ]),
)