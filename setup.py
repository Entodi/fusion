import setuptools

#with open("README.md", "r") as fh:
    #long_description = fh.read()

setuptools.setup(
    name="fusion", # Replace with your own username
    version="0.1",
    author="Alex Fedorov",
    author_email="eidos92@gmail.com",
    description="A multi-modal representation learning package",
    #long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Entodi/fusion",
    packages=setuptools.find_packages(),
)