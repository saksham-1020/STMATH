from setuptools import setup, find_packages

setup(
    name='stmath',
    version='1.0.15',
    author='Saksham Tomar',
    description='Unified Zero-Wrapper Math & AI Framework',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='math ai ml quantum crypto optimization vision stmath',
)
