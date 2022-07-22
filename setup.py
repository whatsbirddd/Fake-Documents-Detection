#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5

from distutils.core import setup

setup(
    name='add_airush classification',
    version='1.1',
    install_requires=[
        'protobuf==3.10.0',
        'matplotlib==3.1.1',
        'tensorflow==2.6.0',
        'keras==2.6.0',
        'pandas==0.23.4',
        'scikit-learn==0.22',
        'transformers==4.15.0',
        'requests==2.25.1',
        'soynlp==0.0.49',
        'sentence-transformers==2.2.2',
        #'wandb==0.12.21'
    ],
)
