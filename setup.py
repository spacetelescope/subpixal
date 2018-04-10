#!/usr/bin/env python
import os
import subprocess
import sys
import pkgutil
from setuptools import setup, find_packages, Extension
from setuptools.command.test import test as TestCommand
from subprocess import check_call, CalledProcessError

try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'subpixal')
DESCRIPTION = metadata.get('description', 'A package aligning images using '
                           'sub-pixel cross correlation.')
AUTHOR = metadata.get('author', 'Mihai Cara')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://www.stsci.edu/')
LICENSE = metadata.get('license', 'BSD-3-Clause')


if not pkgutil.find_loader('relic'):
    relic_local = os.path.exists('relic')
    relic_submodule = (relic_local and
                       os.path.exists('.gitmodules') and
                       not os.listdir('relic'))
    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/jhunkeler/relic.git'])

        sys.path.insert(1, 'relic')
    except CalledProcessError as e:
        print(e)
        exit(1)

import relic.release  # noqa

version = relic.release.get_info()
if not version.date:
    default_version = metadata.get('version', '')
    default_version_date = metadata.get('version-date', '')
    version = relic.git.GitVersion(
        pep386=default_version,
        short=default_version,
        long=default_version,
        date=default_version_date,
        dirty=True,
        commit='',
        post='-1'
    )
relic.release.write_template(version, PACKAGENAME)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['subpixal/tests']
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name=PACKAGENAME,
    version=version.pep386,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'numpy',
    ],
    tests_require=['pytest'],
    packages=find_packages(),
    cmdclass = {'test': PyTest}
)
