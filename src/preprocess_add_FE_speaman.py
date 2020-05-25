import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config
from utils import setup_logger

def application_preprocessing(df, logger, save_path):
    logger.info('Start Prerpocessing')

    threshold = round(len(df) * config.PERCENT)
    null_cnt = df[[col for col in df.columns if col not in config.unused]].isnull().sum()
    reject_col = null_cnt[null_cnt > threshold].index
    df = df[[col for col in df.columns if col not in reject_col]]
    
    #kaggle FE
    logger.info('add new features {}, {}'.format('NEW_CREDIT_TO_ANNUITY_RATIO', 'NEW_CREDIT_TO_GOODS_RATIO'))
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'] 
    #kaggle FE
    logger.info('car to birth ratio')
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    #kaggle FE
    logger.info('credit income ratio')
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    #kaggle FE
    logger.info('new income by organization')
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    #kaggle FE (あまり効かなかった)
    '''
    logger.info('birth to employ ratio')
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    '''

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

    logger.info('Start feature selection by RFE (100 features)')
    train_tmp = df[~df['TARGET'].isnull()]
    test_tmp = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1)

    X = train_tmp[[col for col in train_tmp.columns if col not in config.unused]]
    y = train_tmp.TARGET

    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=100, step=0.5)
    rfe.fit(X, y)
    use_cols = X.columns.values[rfe.support_]

    logger.info('use_cols shape: {}'.format(len(use_cols)))

    rfeData = pd.DataFrame(rfe.transform(X), columns=use_cols)
    train = pd.concat([rfeData, train_tmp.TARGET], axis=1)
    test = test_tmp[use_cols]

    logger.info('train shape: {}, test shape: {}'.format(train.shape, test.shape))

    #logger.info('application shape:{0}'.format(df.shape))
    #logger.info('Save data to directory {0}'.format(save_path))

    #train = df[~df['TARGET'].isnull()]
    #test = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1)
    #train.to_pickle(os.path.join(save_path, 'application_train.pickle'))
    #test.to_pickle(os.path.join(save_path, 'application_test.pickle'))

    logger.info('Finish Preprocessing')

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logger = setup_logger('./logs/preprocessing_{0}.log'.format(NOW))
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_train.csv'), nrows=None)
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_test.csv'), nrows=None)
    all_df = pd.concat([train_df, test_df])
    application_preprocessing(all_df, logger, config.SAVE_PATH)