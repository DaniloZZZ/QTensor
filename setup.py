import setuptools

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = [
    'numpy>=1.18.1',
    'networkx>=2.3',
    'matplotlib>=3.1.3'
    ,'google-api-core[grpc]<=1.14.0'
    ,'cirq'
    ,'qiskit'
    ,'pyrofiler>=0.1.5'
    ,'loguru'
    ,'tqdm'
    ,'click'
    ,'qtensor-qtree'
    ,'lazy-import'
    ,'pynauty-nice'
    ,'sarge'
    ,'cartesian-explorer'

]

setuptools.setup(name='qtensor',
                 version='0.1.2',
                 description='Framework for efficient quantum circuit simulations',
                 url='https://github.com/danlkv/qtensor',
                 keywords='quantum_circuit quantum_algorithms',
                 author='D. Lykov, et al.',
                 author_email='dan@qtensor.org',
                 license='Apache',
                 packages=setuptools.find_packages(),
                 install_requires=REQUIRED_PACKAGES,
                 extras_require={
                     'tensorflow': ['tensorflow<=1.15'],
                 },
                 include_package_data=True,
                 zip_safe=False)
