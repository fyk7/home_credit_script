import os, pickle
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from utils import setup_logger, ModelFactory

def train(X_train, y_train, model_config, logger):
    model = ModelFactory(config.MODEL_NAME, model_config, logger)
    model.fit(X_train, y_train)

    return model

def valid(model, X_test, y_test):
    pred = model.predict(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, pred)
    return auc_score

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/train_{0}.log'.format(NOW))
    #dataframeとして読み込める方が何かと都合が良い。
    df = pd.read_pickle(os.path.join(config.SAVE_PATH, 'application_train.pickle'))
    logger.info('train_df shape: {0}'.format(df.shape))
    X = df[[col for col in df.columns if col not in config.unused]]
    y = df.TARGET

    scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logger.info('Start Kfold validation')
    #stratifiedをsplitする際にはその性質上yも指定しなければならない。
    for i, (train_idx, test_idx) in enumerate(tqdm(kfold.split(X, y))):
        model = train(X.iloc[train_idx], y.iloc[train_idx], config.MODEL_CONFIG, logger)
        auc_score = valid(model, X.iloc[test_idx], y.iloc[test_idx])
        scores.append(auc_score)
        logger.info('Iteration number: {}, AUC Score: {}'.format(i, auc_score))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)
    score_avg = sum(scores)/5

    #model = train(X_train, y_train, config.MODEL_CONFIG, logger)
    #auc_score = valid(model, X_test, y_test)
    logger.info('Average AUC Score: {0}'.format(score_avg))
    logger.info('Save Model to directory {0}'.format(config.SAVE_PATH))

    pickle.dump(model, open(os.path.join(config.SAVE_PATH, config.MODEL_FILE), 'wb'))