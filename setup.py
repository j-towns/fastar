from setuptools import setup, find_packages

setup(
    name='fastar',
    version='0.0',
    author='Jamie Townsend and Julius Kunze',
    author_email='jamiehntownsend@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'jax', 'numba'],
    url='https://github.com/j-towns/fastar',
    license='MIT',
)
