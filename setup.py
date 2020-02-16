# To make majelan_tools as a real python module that can be installed via pip

from setuptools import setup, find_packages
from ada_hub import __version__

setup(name='adaptative_huber_regression',
      version=__version__,
      description='Adaptative Huber Regression',
      authors=['Corentin Ambroise', 'Luis Montero'],
      authors_email=['corentin.ambroise@polytechnique.edu', 'luis.montero@polytechnique.edu'],
      url="https://github.com/fd0r/adaptative_huber_regression.git",  # TODO: Add link to repository
      packages=find_packages(),
      install_requires=["numpy", "sklearn"],
      extra_requires={"scripts": ["tqdm", "matplotlib"]},
      include_package_data=True
      )
