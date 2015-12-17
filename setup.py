try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='Pypsy',
    version='0.1.0',
    packages=['Pypsy'],
    install_requires=[
        'numpy',
        'scipy'
    ],
    url='http://www.musicsensorsemotion.com',
    license='MIT',
    author='Brennon Bortz',
    author_email='brennon@vt.edu',
    description='Electrodermal activity processing and analysis'
)
