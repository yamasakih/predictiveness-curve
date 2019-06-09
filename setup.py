from setuptools import setup

from predictiveness_curve import __version__


long_description = """
Predictiveness curve is a method to display two graphs simultaneously. In both
figures, the x-axis is risk percentile, the y-axis of one figure is the value
of risk, and the y-axis of the other figure is true positive fractions. This
makes it possible to visualize whether the model of risk fits in the medical
field and which value of risk should be used as the basis for the model.
See Am. J. Epidemiol. 2008; 167:362–368 for details.
"""

setup(
    name='predictiveness-curve',
    version=__version__,
    url='https://github.com/yamasakih/predictiveness-curve',
    license='MIT',
    py_modules=['predictiveness_curve'],
    python_requires='>=3.6',
    author='Hiroyuki Yamasaki',
    author_email='yamasaki.phone@gmail.com',
    install_requires=['matplotlib'],
    description='Plot predictiveness curve',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: MIT License',]
)
