"""
This kernel provides a simple starter framework for a LightGBM model.

There are two supplementary functions designed with room to grow as the kernel develops:
    - feature_engineering: Contains the appending of extra features to the main dataset. There are
      a lot of datasets to go through in this challenge, so this is very much in progress
    - process_dataframe: Takes the engineered dataframe and makes it ready for LightGBM. Currently
      is only label encoding thanks to LightGBMs flexbility with nulls and not needing one-hots
"""
import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

""" Load and process inputs """
input_dir = os.path.join('inputdata')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

sample_size = None
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size)
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size)
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size)
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size)
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size)
print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))

# def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
#                         pos_cash_df, prev_app_df, install_df):
#     """ Process the dataframes into a single one containing all the features """
#
#     print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
#
#     # Previous applications
#     agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
#     prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
#     prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
#     merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')
#
#     # Average the rest of the previous app data
#     prev_apps_avg = prev_app_df.groupby('SK_ID_CURR').mean()
#     merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
#                                 how='left', suffixes=['', '_PAVG'])
#     print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
#     # Previous app categorical features
#     prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
#     prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
#                              .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
#     merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
#                             how='left', suffixes=['', '_BAVG'])
#     print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))
#
#     # Credit card data - numerical features
#     wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
#     credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)
#     merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
#                                 how='left', suffixes=['', '_CCAVG'])
#     # Credit card data - categorical features
#     most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
#     cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
#     merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
#                        how='left', suffixes=['', '_CCAVG'])
#     print('Shape after merging with credit card data = {}'.format(merged_df.shape))
#
#     # Credit bureau data - numerical features
#     credit_bureau_avgs = bureau_df.groupby('SK_ID_CURR').mean()
#     merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
#                                 how='left', suffixes=['', '_BAVG'])
#     print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))
#
#     # Pos cash data - weight values by recency when averaging
#     wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
#     f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
#     cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
#                                                  'SK_DPD', 'SK_DPD_DEF'].agg(f)
#     merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
#                                 how='left', suffixes=['', '_CAVG'])
#     # Pos cash data data - categorical features
#     most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
#     cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
#     merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
#                        how='left', suffixes=['', '_CAVG'])
#     print('Shape after merging with pos cash data = {}'.format(merged_df.shape))
#
#     # Installments data
#     ins_avg = install_df.groupby('SK_ID_CURR').mean()
#     merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
#                                 how='left', suffixes=['', '_IAVG'])
#     print('Shape after merging with installments data = {}'.format(merged_df.shape))
#
#     return merged_df

def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder
    print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict

# Merge the datasets into a single one for training
len_train = len(app_train_df)
app_both = pd.concat([app_train_df, app_test_df])
# merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,
#                                 pos_cash_df, prev_app_df, install_df)


merged_df = pd.read_csv('out_merged_df.csv');




# Separate metadata
meta_cols = ['SK_ID_CURR']
meta_df = merged_df[meta_cols]
merged_df.drop(columns=meta_cols, inplace=True)                # 删除用户标号这一列

# Process the data set.
merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)       # 对特征进行处理，使得转换为lightbgm能用的格式

# Re-separate into train and test
train_df = merged_df[:len_train]                                     # 对数据处理后重新选取训练集
test_df = merged_df[len_train:]                                      # 对数据处理后重新选取测试集
del merged_df, app_test_df, bureau_df, bureau_balance_df, credit_card_df, pos_cash_df, prev_app_df
gc.collect()

""" Train the model """
target = train_df.pop('TARGET')                                      # 移除标签数据，并且返回‘Target’的值
test_df.drop(columns='TARGET', inplace=True)                         # 测试数据删除‘TARGET这一列’

# 数据转换
lgbm_train = lgbm.Dataset(data=train_df,
                          label=target,
                          categorical_feature=categorical_feats,
                          free_raw_data=False)
del app_train_df
gc.collect()
# 设置参数
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.1,
    'min_data_in_leaf': 30,                                          # 调大它的值可以防止过拟合，它的值通常设置的比较大。
    'num_leaves': 50,                                                # 决策树叶子节点数 Lightgbm用决策树叶子节点数来确定树的复杂度
    'max_depth': -1,
    'feature_fraction': 0.5,
    'bagging_fraction':0.5,
    'scale_pos_weight': 2,
    'drop_rate': 0.02,
    'max_bin':255,
    'bagging_freq':0,
    'lambda_l1':0,
    'lambda_l2':0,
    'min_split_gain':0
}

# 网格搜索，寻找最佳的超参数
best_params = {}
max_auc = float('-inf')
# 准确率
print('调参1：提高准确率')
for num_leaves in range(20,200,5):
    for max_depth in range(3,8,1):
        lgbm_params['num_leaves'] = num_leaves
        lgbm_params['max_depth'] = max_depth
        # 使用给定的参数完成交叉验证
        cv_results = lgbm.cv(params=lgbm_params,
                             train_set=lgbm_train,
                             seed=2018,
                             nfold=5,
                             metrics=['auc'],
                             early_stopping_rounds=50,
                             verbose_eval=50,
                             num_boost_round=2000,
        )
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

lgbm_params['num_leaves'] = best_params['num_leaves']
lgbm_params['max_depth'] = best_params['max_depth']


# 过拟合
print('调参2：降低过拟合')
for max_bin in range(1,255,5):
    for min_data_in_leaf in range(10,200,5):
        lgbm_params['max_bin'] = max_bin
        lgbm_params['min_data_in_leaf'] = min_data_in_leaf

        cv_results = lgbm.cv(params=lgbm_params,
                             train_set=lgbm_train,
                             seed=42,
                             nfold=5,
                             metrics=['auc'],
                             early_stopping_rounds=50,
                             num_boost_round=2000,
                             verbose_eval=50)
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).argmax()

        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf
lgbm_params['max_bin'] = best_params['max_bin']
lgbm_params['min_data_in_leaf'] = best_params['min_data_in_leaf']

# 继续降低过拟合
print("调参3：降低过拟合")
for feature_fraction in [0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for bagging_fraction in [0.0, 0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        for bagging_freq in range(0,50,5):
            lgbm_params['feature_fraction'] = feature_fraction
            lgbm_params['bagging_fraction'] = bagging_fraction
            lgbm_params['bagging_freq'] = bagging_freq

            cv_results = lgbm.cv(params=lgbm_params,
                                 train_set= lgbm_train,
                                 nfold=5,
                                 seed=42,
                                 metrics=['auc'],
                                 early_stopping_rounds=50,
                                 num_boost_round=2000,
                                 verbose_eval=50)
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmax()

            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

lgbm_params['feature_fraction'] = best_params['feature_fraction']
lgbm_params['bagging_fraction'] = best_params['bagging_fraction']
lgbm_params['bagging_freq'] = best_params['bagging_freq']

print("调参4：降低过拟合")
for lambda_l1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for lambda_l2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            lgbm_params['lambda_l1'] = lambda_l1
            lgbm_params['lambda_l2'] = lambda_l2
            lgbm_params['min_split_gain'] = min_split_gain

            cv_results = lgbm.cv(
                params=lgbm_params,
                train_set=lgbm_train,
                seed=42,
                nfold=5,
                metrics=['auc'],
                early_stopping_rounds=50,
                num_boost_round=2000,
                verbose_eval=50
            )

            mean_auc = pd.Series(cv_results['auc-mean']).min()
            boost_rounds = pd.Series(cv_results['auc-mean']).argmin()

            if mean_auc > max_auc:
                max_auc = mean_auc
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
                best_params['min_split_gain'] = min_split_gain

lgbm_params['lambda_l1'] = best_params['lambda_l1']
lgbm_params['lambda_l2'] = best_params['lambda_l2']
lgbm_params['min_split_gain'] = best_params['min_split_gain']

print(best_params)









#
#
# cv_results = lgbm.cv(train_set=lgbm_train,
#                      params=lgbm_params,
#                      nfold=5,
#                      num_boost_round=2000,
#                      early_stopping_rounds=50,
#                      verbose_eval=50,
#                      metrics=['auc'])

optimum_boost_rounds = np.argmax(cv_results['auc-mean'])             # 返回auc-mean的最大值的索引的位置
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))      # 打印出该值
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))  # 打印出auc-mean的最大值
# 开始使用lightbgm训练模型
clf = lgbm.train(train_set=lgbm_train,
                 params=lgbm_params,
                 num_boost_round=optimum_boost_rounds)

""" Predict on test set and create submission """
y_pred = clf.predict(test_df)                                        # 在测试集上进行预测
out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': y_pred}) # 输出用户名和预测到的标签值
out_df.to_csv('submission.csv', index=False)                                               # 将输出转换为.csv文件
fig, ax = plt.subplots(1, 1, figsize=[5, 7])
lgbm.plot_importance(clf, ax=ax, max_num_features=30)                                      # 画出经过lgbm算法后排序得到的最重要的20个特征
plt.savefig('feature_importance.png')
