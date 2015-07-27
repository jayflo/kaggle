## Installation

- install python and dependencies [scikit-learn](http://scikit-learn.org/stable/index.html)
- `git clone https://github.com/jayflo/kaggle.git`
- add `/kcmn` to `PYTHONPATH` environment variable
- install competition data to `/data` directory.  See how data is loaded
in each competition's `main.py` file to determine exact `/data` structure, e.g.

  ```
  pd.read_csv('../data/titanic/train.csv'),
  pd.read_csv('../data/titanic/test.csv')
  ```

in `/titanic/main.py` means that competition expects the training data (from
[kaggle](www.kaggle.com)) to reside in `/data/titanic/train.csv` (and similarly
for the test data).
