import os
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import config
from utils import setup_logger

#下のloggerにはsetup_loggerの戻り値を入れる。
def application_preprocessing(df, logger, save_path):
    logger.info('Start Prerpocessing')

    threshold = round(len(df) * config.PERCENT)
    null_cnt = df[[col for col in df.columns if col not in config.unused]].isnull().sum()
    #rejectするカラムのリストをまずは作成。
    reject_col = null_cnt[null_cnt > threshold].index
    df = df[[col for col in df.columns if col not in reject_col]]

    #数値型のカラムかそうで無いかはdtypeがobjectであるかそうで無いかで判断。
    num_col = [col for col in df.columns if df[col].dtype != 'object' and col not in config.unused]
    not_num_col = [col for col in df.columns if col not in num_col and col not in config.unused]

    df = pd.get_dummies(df, columns=not_num_col, dummy_na=True)

    #loggerで処理を要件定義書のように記述するのも良いかもしれない。
    logger.info('Handling Missing Values')
    for col in df.columns:
        if col in num_col:
            df[col].fillna(df[col].mean(), inplace=True)
            if config.SCALING:
                sc = StandardScaler()
                #各カラムごとに処理していることに注意、fit_transformは二次元配列にのみ対応しているから、reshape必須
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

    #下の方でconcatした時にtestではTARGETカラムがnullになる。
    #train = df[~df['TARGET'].isnull()]
    #test = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1)
    train.to_pickle(os.path.join(save_path, 'application_train.pickle'))
    test.to_pickle(os.path.join(save_path, 'application_test.pickle'))

    logger.info('Finish Preprocessing')

if __name__ == '__main__':
    NOW = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    #ここでloggerを定義している。logを吐き出す先のfileパスを指定する。
    logger = setup_logger('./logs/preprocessing_{0}.log'.format(NOW))
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_train.csv'), nrows=None)
    test_df = pd.read_csv(os.path.join(config.DATA_PATH, 'application_test.csv'), nrows=None)
    #TARGETがあるものと無いものを縦にconcatしたらどうなるのかを調べる。（testの方のTARGETが０になるだけ？）
    #TARGETが0になるだけで合っている。上の~df['TARGET'].isnull()というコードから逆算して考えれば
    all_df = pd.concat([train_df, test_df])
    application_preprocessing(all_df, logger, config.SAVE_PATH)