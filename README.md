## Installation

- install python and dependencies, i.e. [scikit-learn](http://scikit-learn.org/stable/index.html)
- `git clone https://github.com/jayflo/kaggle.git`
- add `/mlcmn` to `PYTHONPATH` environment variable
- install competition data to `/data` directory.  See how data is loaded
in each competition's `main.py` file to determine exact `/data` structure, e.g.

  ```
  pd.read_csv('./data/train.csv'),
  pd.read_csv('./data/test.csv')
  ```

  in `/titanic/main.py` means that competition expects the training data (from
  [kaggle](www.kaggle.com)) to reside in `/titanic/data/train.csv` (and similarly
  for the test data).
