from setuptools import setup, find_packages

setup(name='TractSeg',
        version='2.1',
        description='Fast and accurate segmentation of white matter bundles',
        url='https://github.com/MIC-DKFZ/TractSeg/',
        author='Jakob Wasserthal',
        author_email='j.wasserthal@dkfz-heidelberg.de',
        python_requires='>=2.7',
        license='Apache 2.0',
        packages=find_packages(),
        install_requires=[
            'future',
            'numpy',
            'nibabel>=2.3.0',
            'matplotlib',
            'sklearn',
            'scipy',
            'tqdm',
            'six',
            'psutil',
            'dipy>=1.0.0',
            'joblib>=0.13.2'
            # 'batchgenerators==0.17'   #results in error (version...)
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
            'bin/TractSeg', 'bin/ExpRunner', 'bin/flip_peaks', 'bin/calc_FA', 'bin/Tractometry',
            'bin/download_all_pretrained_weights', 'bin/Tracking', 'bin/rotate_bvecs',
            'bin/plot_tractometry_results'
        ],
        package_data = {'tractseg.resources': ['MNI_FA_template.nii.gz',
                                      'random_forest_peak_orientation_detection.pkl']},
    )
