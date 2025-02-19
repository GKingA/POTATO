from setuptools import find_packages, setup




setup(
    name="xpotato",
    version="0.1.4",
    description="XAI human-in-the-loop information extraction framework",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adaamko/POTATO",
    author="Adam Kovacs, Gabor Recski",
    author_email="adam.kovacs@tuwien.ac.at, gabor.recski@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "pandas >= 1.3.5",
        "tqdm",
        "stanza == 1.3.0",
        "scikit-learn == 1.0.2",
        "eli5 == 0.11.0",
        "jinja2 == 3.0.1",
        "graphviz == 0.18.2",
        "penman >= 1.2.1",
        "networkx == 2.6.3",
        "rank_bm25 == 0.2.1",
        "streamlit == 1.3.1",
        "streamlit-aggrid == 0.2.3.post2",
        "scikit-criteria == 0.5",
        "tuw-nlp == 0.0.9",
        "amrlib == 0.6.0",
        "protobuf==3.20.0",
        "pytest >= 7.1.3",
        "fastapi",
        "uvicorn[standard]"
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
