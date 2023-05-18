from setuptools import find_packages, setup

setup(
    name="data_engineer_answers",
    packages=find_packages(exclude=["data_engineer_answers_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "dagster-docker",
        "pandas",
        "pyarrow",
        "joblib",
        "scikit-learn",
        "flask",
        "kaggle",
        "python-dotenv",
        "pytest-dotenv"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)
