from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sl3",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Driven Hedge Fund Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sl3",  # Replace with your repository URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "yfinance",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "statsmodels",
        "cvxopt",
        "cvxpy",
        "requests",
        "tqdm",
        "streamlit",
        "dash",
        "dash-core-components",
        "dash-html-components",
        "pytest",
        "python-dotenv",
        "PyYAML"
    ],
    entry_points={
        'console_scripts': [
            'sl3=main:main',  # If you want to create a command-line tool
        ],
    },
)

# Example Usage:
# 1.  Create a setup.py file in the root directory of your project.
# 2.  Run `python setup.py sdist bdist_wheel` to build the package.
# 3.  Run `pip install dist/sl3-0.1.0-py3-none-any.whl` to install the package.
# 4.  Run `sl3` from the command line (if you defined an entry point).