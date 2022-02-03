from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension('tractseg.libs.tractseg_prob_tracking',
            sources=['tractseg/libs/tractseg_prob_tracking.pyx'],
            include_dirs=[numpy.get_include()],
        )
]

setup(name='TractSeg',
        version='2.4',
        description='Fast and accurate segmentation of white matter bundles',
        long_description="See Readme.md on github for more details.",
        url='https://github.com/MIC-DKFZ/TractSeg/',
        author='Jakob Wasserthal',
        author_email='j.wasserthal@dkfz-heidelberg.de',
        python_requires='>=3.5',
        license='Apache 2.0',
        packages=find_packages(),
        ext_modules=cythonize(ext_modules),
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
            'fury',
            'joblib>=0.13.2',
            'seaborn'
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
            'bin/plot_tractometry_results', 'bin/get_image_spacing', 'bin/remove_negative_values'
        ],
        package_data = {'tractseg.resources': ['MNI_FA_template.nii.gz',
                                      'random_forest_peak_orientation_detection.pkl']},
    )
