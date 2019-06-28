from setuptools import setup

__version__ = '0.2.1'

with open("README.md") as f:
    long_description = f.read()

setup(
    name='predictiveness-curve',
    version=__version__,
    url='https://github.com/yamasakih/predictiveness-curve',
    license='MIT',
    py_modules=['predictiveness_curve'],
    python_requires='>=3.6',
    author='Hiroyuki Yamasaki',
    author_email='yamasaki.phone@gmail.com',
    install_requires=['matplotlib', 'numpy'],
    description='Plot predictiveness curve',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: MIT License',]
)
