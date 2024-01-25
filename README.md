# Modeling the ADORE 't-F2F' challenge

After generating the ADORE dataset, we first tackled the modeling of the challenge on the taxonomic group of fish 't-F2F' to determine the potential and limitations of applying machine learning models to the predictions of ecotoxicological outcomes.


## A. Project description

The MLTox project ([MLTox: Enhancing Toxicological Testing through Machine Learning](https://www.datascience.ch/projects/mltox)) is a collaborative project between the Swiss Data Science Center and the environmental toxicology (UTOX) department and the department of systems analysis, integrated assessment and modeling (SIAM) at eawag.

As a first step, we compiled the [ADORE benchmark dataset](TODO), which contains several challenges of varying complexity. The 't-F2F' challenge was selected to be modeled first. It contains 26k entries on 140 fish species and 1,905 chemicals. We modeled this data using LASSO, random forest, XGBoost, and GP regression.

We have summarized our work in the paper **Machine learning-based prediction of fish acute mortality: Implementation,  interpretation, and regulatory relevance**. This repository contains all the code produce the results. 



## B. Getting started

First, the repository needs to be setup.

1. Clone the repository.

This step needs an ssh connection.

```
git clone git@gitlab.renkulab.io:mltox/adore.git
cd adore/
```

2. Install git LFS.

The data files are stored as Large File Storage (LFS) files. This step can take a few minutes.

```
git lfs install --local
git lfs pull -I "data/raw/*"
```

3. Create a conda environment.

This command installs the environment directly in the project folder using the provided `environment.yml`.

```
conda env create --prefix ./conda-env --file ./environment.yml
conda activate ./conda-env
```

OR

If you prefer mamba:

```
mamba env create --prefix ./conda-env --file ./environment.yml
conda activate ./conda-env
```

4. View the dataset.

Open your favourite IDE and run the `10_view-a-dataset.py` script.

OR 

Run the script directly in the conda environment:

```
python 10_view-a-dataset.py
```



## C. Example / Usage

To produce our results, several scripts are needed. 

1. Hyperparameter tuning and cross-validation per model: scripts 11, 12, 13, 14
2. Evaluate cross-validation per model: scripts 21, 22, 23, 24
3. Generate test results per model: scripts 31, 32, 33, 34
4. Evaluate all results: script 46
5. Evaluate residuals: script 47
6. Evaluate macro-averages: script 48
7. Evaluate feature importances and generate species-sensitivity distributions (SSD) for random forest and XGBoost: script 53-54

The scripts start with a two-digit number. GP models are named x1, LASSO x2, XGBoost, x3, and RF x4. If the number is followed by an 'a', only the top3 features are considered for modeling. If the number is followed by a 'b', single species challenges are modeled (not part of the paper).

