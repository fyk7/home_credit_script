SEED = 2019
DATA_PATH = './input'
SAVE_PATH = './output'
MODEL_NAME = 'LogisticRegression'
MODEL_FILE = '{0}_model.pickle'.format(MODEL_NAME)
TEST_SIZE = 0.2
PERCENT = 1 #0.5
SCALING = True
unused = ['SK_ID_CURR', 'TARGET']

MODEL_CONFIG = {
    'LogisticRegression': {
        'C': 1.0,
        'random_state': SEED,
        'max_iter': 100,
        'penalty': 'l2',
        'n_jobs': -1,
        'solver': 'lbfgs',
        #'class_weight': {0:1, 1:2},
    },
    'RandomForest': {
        'max_depth': 8,
        'min_sample_split': 2,
        'n_estimator': 200,
        'random_state': SEED,
        'class_weight': {0:1, 1:2},
    }
}

#MODEL_CONFIG = {logistic:, randomfore:}の構造になっている。