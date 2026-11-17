# NYC Airbnb - how much should a listing charge per night?

Semester project for the Artificial Neural Networks and Deep Learning course, by **Gustavo Cruz, Henrique Mayor and Pedro Fardin**. We use the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (48,895 listings, 2019 snapshot) and the task is **regression on `price`**.

The project has a few deliverables and each one gets its own page:

| | deliverable | status |
|---|---|---|
| 1 | [Exploratory Data Analysis](projects/eda/index.md) - data dictionary, quality, distributions, correlations and the preprocessing pipeline | done |
| 2 | [Regression with a neural network](projects/regression/index.md) - baselines, MLP in PyTorch, regularisation, hyperparameter search, test evaluation over 3 seeds and error analysis | done |

## Running the code

```bash
git clone https://github.com/henriquessm/nyc-airbnb-project.git
cd nyc-airbnb-project
pip install -r requirements.txt
python code/eda.py          # regenerates every figure and docs/projects/eda/results.json
python code/preprocess.py   # prints the pipeline sanity report
python code/regression.py   # trains the baselines and the network, figures + docs/projects/regression/results.json
```

The dataset is committed under `data/` so a clean clone runs without a Kaggle account.
