from setuptools import setup, find_packages

setup(
    name='riverpy',
    version='0.1.1',
    packages=find_packages(), # organise the internal dependencies, not external 
    install_requires=[
        'autologging',
    ],    
    description='',
    author='Kajetan Chrapkiewicz',
    author_email='k.chrapkiewicz17@imperial.ac.uk',
    url='https://github.com/kmch/RiverPy',
)
