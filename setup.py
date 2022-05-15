"""Install nlpsharpness"""
import os
import setuptools

def setup_package():
  long_description = "Sharpness methods for Huggingface transformers"
  setuptools.setup(
      name='nlpsharpness',
      version='0.0.1',
      description='Sharpness methods for Huggingface transformers',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Asa Cooper Stickland',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      dependency_links=[
          'https://download.pytorch.org/whl/torch_stable.html',
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7.10',
      ],
      keywords='text nlp machinelearning',
      install_requires=[
        'datasets==1.18.2',
        'scikit-learn==0.24.2',
        'tensorboard==2.5.0',
        'matplotlib==3.4.2',
        'torch==1.11.0',
        'transformers==4.16.2'
      ],
  )


if __name__ == '__main__':
  setup_package()
