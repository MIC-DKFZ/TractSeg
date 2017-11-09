from setuptools import setup, find_packages

setup(name='TractSeg',
        version='0.5',
        description='Fast and accurate segmentation of white matter bundles',
        url='todo',
        author='Jakob Wasserthal',
        author_email='j.wasserthal@dkfz-heidelberg.de',
        python_requires='>=2.6, <3',
        license='Apache 2.0',
        packages=find_packages(),
        #Torch/Lasagne has to be installed manually
        install_requires=[
            'numpy',
            'nibabel',
            'torch',
            'matplotlib',
            'sklearn',
            'scipy',
            'tqdm'
        ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            'TractSeg.py', 'ExpRunner.py'   #todo: can we remove file ending?
        ],
      )

#https://stackoverflow.com/questions/8247605/configuring-so-that-pip-install-can-work-from-github
