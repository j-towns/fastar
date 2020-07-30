from setuptools import setup

setup(
    name='fastar',
    version='0.0',
    author='Jamie Townsend and Julius Kunze',
    author_email='jamiehntownsend@gmail.com',
    packages=['fastar'],
    install_requires=['numpy', 'jax', 'numba'],
    url='https://github.com/j-towns/fastar',
    license='MIT',
)
