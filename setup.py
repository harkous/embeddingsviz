from setuptools import setup

setup(name='embeddingsviz',
      version='0.1',
      description='Visualize Embeddings of a Vocabulary in TensorBoard, Including the Neighbors',
      classifiers=[
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic',
      ],
      url='http://github.com/harkous/embeddingviz',
      author='Hamza Harkous',
      license='MIT',
      packages=['embeddingsviz'],
      install_requires=[
          "tensorflow",
          "numpy",
      ],
      zip_safe=False)