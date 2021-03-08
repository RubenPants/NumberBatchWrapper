"""Setup the package."""
import pathlib

from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

INSTALL_REQUIRES = [
]

# noinspection SpellCheckingInspection
setup(
        name="number_batch_wrapper",
        version="0.1.0",
        description="Wrapper around the ConceptNet Number Batch library to improve performance.",
        long_description=README,
        long_description_content_type="text/markdown",
        url="https://github.com/RubenPants/NumberBatchWrapper",
        author="RubenPants",
        author_email="broekxruben@gmail.com",
        license="LICENSE",
        classifiers=["Programming Language :: Python :: 3", "Programming Language :: Python :: 3.8", ],
        packages=find_packages(exclude=("demos", "img", "tests", "notebooks", "doc", "scripts")),
        include_package_data=True,
        package_data={"": ["data/synonym_config.pkl"]},
        install_requires=INSTALL_REQUIRES,
)
