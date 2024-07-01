# Feature Selection Problem

This project focuses on optimizing feature selection for predicting housing prices within the Airbnb business model.

## Instructions

The dataset can be obtained from: [Kaggle Singapore Airbnb Dataset](https://www.kaggle.com/datasets/jojoker/singapore-airbnb)

To test the loss functions, optimization algorithms, and regression models, simply clone this repository. Consider installing the requirements from `requirements.txt`.
```bash
pip install -r requirements.txt
```

The `test_script.py` module will execute all combinations of algorithms and models, and will also run each combination with 10 different seeds.

To test a specific combination of an experiment, you need to first instantiate an `Evaluator`, specifying the path of the CSV, the type of encoding (`drop`, `target`, or `label`) for categorical features, the regression model (`linear`, `lasso`, or `ridge`), and the loss function (`rmse`, `r2`, or `combined`):

```python
Evaluator(data_path, encoder, regression_model, loss_function)
```

##Notes
- The index_col='id' option works for the CSV available in the repository. However, you should consider modifying it if you plan to use another dataset.
