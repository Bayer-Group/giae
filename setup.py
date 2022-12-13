from setuptools import setup, find_packages

setup(
   name='giae',
   version='0.1',
   description='Research code of the paper "Unsupervised Learning of Group Invariant and Equivariant Representations" by Winter et al. 2022',
   author='Robin Winter, Marco Bertolini, Tuan Le',
   author_email='',
   packages=find_packages(), install_requires=['torch', 'numpy', 'pytorch_lightning', 'e2cnn', 'torch_geometric']
  )
