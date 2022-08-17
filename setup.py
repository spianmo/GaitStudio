from setuptools import setup

setup(
    name='HealBone-GaitAnalysis',
    version='0.0.1',
    packages=[''],
    url='https://github.com/spianmo',
    license='MIT',
    python_requires=">=3",
    install_requires=["mediapipe", "opencv-python", "scipy"],
    author='Finger',
    author_email='finger@spianmo.com',
    description='Real-Time Gait Cycle Analysis'
)
