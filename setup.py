from setuptools import setup, find_packages

setup(
    name='diabetes_prediction',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'keras',
        'matplotlib',
        'tensorflow',
        'gunicorn',
    ],
    entry_points={
        'console_scripts': [
            'start-app = app:app.run',
        ],
    },
    author='Badreenath Gudipudi',
    author_email='badreenathgudipudi@gmail.com',
    description='A deep learning project for diabetes prediction using a CNN model',
    long_description='This project includes preprocessing, model training, evaluation, and prediction for Eye disease detection using a dataset of retina scan images.',
    long_description_content_type='text/markdown',
    url="https://github.com/badrinath0526/Eye-disease-detection-using-CNN",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)