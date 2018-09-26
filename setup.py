from setuptools import setup, find_packages

setup(name='TractSeg',
        version='1.5',
        description='Fast and accurate segmentation of white matter bundles',
        url='https://github.com/MIC-DKFZ/TractSeg/',
        author='Jakob Wasserthal',
        author_email='j.wasserthal@dkfz-heidelberg.de',
        python_requires='>=2.7',
        license='Apache 2.0',
        packages=find_packages(),
        #Torch/Lasagne has to be installed manually
        install_requires=[
            'numpy',
            'nibabel>=2.3.0',
            'matplotlib',
            'sklearn',
            'scipy',
            'tqdm',
            'six',
            'psutil',
            'dipy'
            # 'batchgenerators==0.17'   #results in error (version...)
        ],
        # dependency_links=[
        #     'https://github.com/MIC-DKFZ/batchgenerators/archive/tractseg_stable.zip#egg=batchgenerators-0.17'
        # ],
        zip_safe=False,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
        ],
        scripts=[
            'bin/TractSeg', 'bin/ExpRunner', 'bin/flip_peaks', 'bin/calc_FA', 'bin/Tractometry'
        ],
      )
