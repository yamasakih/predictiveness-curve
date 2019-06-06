from setuptools import setup

#from predictiveness_curve import __version__


long_description=''

setup(
    name='predictiveness_curve',
    version='0.1.0',#__version__,
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
    keywords='',
    classifiers=['License :: OSI Approved :: MIT License',]
)

