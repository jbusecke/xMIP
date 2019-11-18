from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='cmip6_preprocessing',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Some useful functions to make analysis across cmip6 modesl easier",
    license="MIT",
    author="Julius Busecke",
    author_email='jbusecke@princeton.edu',
    url='https://github.com/jbusecke/cmip6_preprocessing',
    packages=['cmip6_preprocessing'],
    
    install_requires=requirements,
    keywords='cmip6_preprocessing',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
