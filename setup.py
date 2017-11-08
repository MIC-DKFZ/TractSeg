from setuptools import setup

setup(name='batchgenerators',
        version='1.0',
        description='Fast and accurate segmentation of white matter bundles',
        url='todo',
        author='Jakob Wasserthal',
        author_email='j.wasserthal@dkfz-heidelberg.de',
        python_requires='>=2.6, <3',
        license='MIT',
        # packages=['batchgenerators', 'batchgenerators.augmentations', 'batchgenerators.generators',
        # 'batchgenerators.examples', 'batchgenerators.transforms', 'batchgenerators.dataloading'],
        requires=[
            'numpy(>=1.6)'
        ],
        zip_safe=False,
        #Add pretrained Weights here ?
        # data_files=[('my_data', ['data/data_file'])],
        #Is this useful?
        # entry_points={
        #     'console_scripts': [
        #         'sample=sample:main',
        #     ],
        # },
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Operating System :: Unix',
            'Operating System :: MacOS'
            # 'Operating System :: Microsoft :: Windows',
        ],
        scripts=[
            'TractSeg'
        ],

      )

#https://stackoverflow.com/questions/8247605/configuring-so-that-pip-install-can-work-from-github