## Test suite developer notes
### Running the tests

#### Additional Dependencies (also listed in `environment.yml`)
- `pytest=9.0.2`

Please run the following in the root of the folder

```
pytest
```

_Note: Minor warnings may appear due to small test dataset or limited hyperparameter grids, but they can be ignored if all tests pass._

Ex.

```
UserWarning: The total space of parameters 2 is smaller than n_iter=10. Running 2 iterations. For exhaustive searches, use GridSearchCV.
```

```
UserWarning: The least populated class in y has only 2 members, which is less than n_splits=3.
```