from setuptools import find_packages
from setuptools import setup


setup(name='keras_anomaly_detection',
      version='0.0.1',
      description='Anomaly Detector',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-anomaly-detection',
      download_url='https://github.com/chen0040/keras-anomaly-detection/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras==2.1.2'],
      packages=find_packages())
