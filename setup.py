import setuptools


setuptools.setup(
    name="smartsnn",
    version="0.0.0.dev1",
    author="Zachary R. Claytor",
    license="MIT",
    python_requires='>=3',
    install_requires=["numpy", "torch", "astropy"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["models/*pt"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
