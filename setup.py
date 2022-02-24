from setuptools import find_packages, setup

setup(
    name="algo_tools",
    version="0.0.5",
    author="zjq",
    author_email="zjq@zjq2133318.com",
    description="best algo tools packages",
    long_description='123456',
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'requests', 'tqdm', 'aiohttp']
)
