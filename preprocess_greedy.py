import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

import config
from utils import setup_logger

def evaluate(features):
    X_eval = df[features]
    y_eval = df.TARGET
    logreg = LogisticRegression()
    logreg.fit(X_eval, y_eval)
    score = 

def application_preprocessing(df, logger, save_path):
    logger.info('Start Prerpocessing')

    threshold = round(len(df) * config.PERCENT)
    null_cnt = df[[col for col in df.columns if col not in config.unused]].isnull().sum()
    reject_col = null_cnt[null_cnt > threshold].index
    df = df[[col for col in df.columns if col not in reject_col]]

    num_col = [col for col in df.columns if df[col].dtype != 'object' and col not in config.unused]
    not_num_col = [col for col in df.columns if col not in num_col and col not in config.unused]

    df = pd.get_dummies(df, columns=not_num_col, dummy_na=True)

    logger.info('Handling Missing Values')
    for col in df.columns:
        if col in num_col:
            df[col].fillna(df[col].mean(), inplace=True)
            if config.SCALING:
                sc = StandardScaler()
                df[col] = sc.fit_transform(df[col].values.reshape(-1, 1))

    logger.info('application shape:{0}'.format(df.shape))
    logger.info('Save data to directory {0}'.format(save_path))

    logger.info('start simple greedy selection')

    best_score = 9999.0
    candidates = np.random.RandomState(42).permutation(df.columns)
    selected = set([])

    for feature in candidates:
        # 特徴量のリストに対して精度を評価するevaluate関数があるものとする
        fs = list(selected) + [feature]
        score = evaluate(fs)

        # スコアは低い方が良いとする
        if score < best_score:
            selected.add(feature)
            best_score = score
            print(f'selected:{feature}')
            print(f'score:{score}')

    logger.info(f'selected features: {selected}')

    train = df[~df['TARGET'].isnull()]
    test = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1)
    train.to_pickle(os.path.join(save_path, 'application_train.pickle'))
    test.to_pickle(os.path.join(save_path, 'application_test.pickle'))

    logger.info('Finish Preprocessing')

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/preprocessing_{0}.log'.format(NOW))
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_train.csv'), nrows=None)
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_test.csv'), nrows=None)
    all_df = pd.concat([train_df, test_df])
    application_preprocessing(all_df, logger, config.SAVE_PATH)