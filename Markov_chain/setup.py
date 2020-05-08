from setuptools import setup

setup(name='Markov_chain',
      version='0.4',
      description='A package that contains functions to compute Markov chain measures.',
      url='http://github.com/jberkhout/Markov_chain',
      author='Joost Berkhout',
      author_email='j.berkhout@cwi.nl',
      license='MIT',
      packages=['Markov_chain'],
      package_data={'Markov_chain': ['data/*.csv']},
      zip_safe=False,
    python_requires='>=3.6')
