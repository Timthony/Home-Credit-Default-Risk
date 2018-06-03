#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 20:28:05 2018

@author: arcstone_mems_108
"""
import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import tensorflow.contrib as contrib

# application_train = pd.read_csv("application_train.csv")
# application_test = pd.read_csv("application_test.csv")
# bureau = pd.read_csv("bureau.csv")
# bureau_balance = pd.read_csv("bureau_balance.csv")
# credit_card_balance = pd.read_csv("credit_card_balance.csv")
# installments_payments = pd.read_csv("installments_payments.csv")
# previous_application = pd.read_csv("previous_application.csv")
# POS_CASH_balance = pd.read_csv("POS_CASH_balance.csv")

#==================================【数据处理模块】=========================================
# 导入inputdata文件夹的数据，并且打印出该目录下的文件名称
input_dir = os.path.join('inputdata')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('正在载入数据集...')

sample_size = None
# 读入8个原始数据集
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


def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
    """ Process the dataframes into a single one containing all the features """

    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))

    # Previous applications
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)                          # 对数据进行分组
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']                        #
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')   # 对数据进行连接

    # Average the rest of the previous app data
    prev_apps_avg = prev_app_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PAVG'])
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))
    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
                             .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CCAVG'])
    # Credit card data -
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    # Credit bureau data - numerical features
    credit_bureau_avgs = bureau_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])
    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    # Installments data
    ins_avg = install_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_IAVG'])
    print('Shape after merging with installments data = {}'.format(merged_df.shape))

    return merged_df



def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()                                      # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))            # 对标签值数字化，空缺部分填充null
        encoder_dict[feat] = encoder                                                     # 将结果存入encoder_dict

    return input_df, categorical_feats.tolist(), encoder_dict                            # 将数组矩阵转换为列表



#
# Merge the datasets into a single one for training
# 将数据集合并成一个来进行训练
len_train = len(app_train_df)                                                            # 训练数据的长度
app_both = pd.concat([app_train_df, app_test_df])                                        # 将数据根据相同的轴融合
merged_df = feature_engineering(app_both, bureau_df, bureau_balance_df, credit_card_df,  # 进行特征工程的处理
                                pos_cash_df, prev_app_df, install_df)

#merged_df.to_csv('out_merged_df.csv', index=False)


out_merged_df = pd.read_csv('out_merged_df.csv')




# Separate metadata
meta_cols = ['SK_ID_CURR']
meta_df = merged_df[meta_cols]
merged_df.drop(columns=meta_cols, inplace=True)

# Process the data set.
merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)



# Extract target before scaling
labels = merged_df.pop('TARGET')
labels = labels[:len_train]

# Reshape (one-hot)
target = np.zeros([len(labels), len(np.unique(labels))])
target[:, 0] = labels == 0
target[:, 1] = labels == 1



dangerous_feats = []
for feat in categorical_feats:
    feat_cardinality = len(merged_df[feat].unique())
    if (feat_cardinality > 10) & (feat_cardinality <= 100):
        print('Careful: {} has {} unique values'.format(feat, feat_cardinality))
    if feat_cardinality > 100:
        categorical_feats.remove(feat)
        dangerous_feats.append(feat)
        print('Dropping feat {} as it has {} unique values'.format(feat, feat_cardinality))
merged_df.drop(columns=dangerous_feats, inplace=True)
merged_df = pd.get_dummies(data=merged_df, columns=categorical_feats)
print('Shape after one-hot encoding = {}'.format(merged_df.shape))





null_counts = merged_df.isnull().sum()
null_counts = null_counts[null_counts > 0]
null_ratios = null_counts / len(merged_df)

# Drop columns over x% null
null_thresh = .8
null_cols = null_ratios[null_ratios > null_thresh].index
merged_df.drop(columns = null_cols, inplace=True)
print('Columns dropped for being over {}% null:'.format(100*null_thresh))
for col in null_cols:
    print(col)

# Fill the rest with the mean (TODO: do something better!)
# merged_df.fillna(merged_df.median(), inplace=True)
merged_df.fillna(0, inplace=True)



scaler = StandardScaler()
merged_df = scaler.fit_transform(merged_df)









# Re-separate into labelled and unlabelled
train_df = merged_df[:len_train]
predict_df = merged_df[len_train:]
del merged_df, app_train_df, app_test_df, bureau_df, bureau_balance_df, credit_card_df, pos_cash_df, prev_app_df
gc.collect()
# 对数据进行分割，训练集和测试集的比例为8：2
# Create a validation set to check training performance
X_train, X_valid, y_train, y_valid = train_test_split(train_df, target, test_size=0.2, random_state=666)


# 定义图的参数Fixed graph parameters
n_inputs = X_train.shape[1]
n_classes = 2
n_hidden_1 = 20
n_hidden_2 = 20
n_hidden_3 = 10
n_hidden_4 = 10

# Learning parameters
learning_rate = 0.001
n_epochs = 30
n_iterations = 1000
batch_size = 250

# # Graph
tf.reset_default_graph()

# Placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None, n_classes), name='labels')

# 定义神经网络的每一层
# 为了防止过拟合，第一层增加了L2正则化去调整权重，和一些dropout层
# Define the network layers.
# Overfitting is a challenge so add L2 regularisation to weights in 1st layer &
# a couple of dropout layers
with tf.name_scope('dnn'):
    hidden_layer_1 = tf.layers.dense(inputs=X,
                                     units=n_hidden_1,
                                     name='first_hidden_layer',
                                     activation=tf.nn.relu,
                                     kernel_regularizer= contrib.layers.l2_regularizer(scale=0.3)
                                     )

    drop_layer_1 = tf.layers.dropout(inputs=hidden_layer_1,
                                     rate=0.5,
                                     name='first_dropout_layer')

    hidden_layer_2 = tf.layers.dense(inputs=drop_layer_1,
                                     units=n_hidden_2,
                                     name='second_hidden_layer',
                                     activation=tf.nn.relu)

    drop_layer_2 = tf.layers.dropout(inputs=hidden_layer_2,
                                     rate=0.5,
                                     name='second_dropout_layer')

    hidden_layer_3 = tf.layers.dense(inputs=drop_layer_2,
                                     units=n_hidden_3,
                                     name='third_hidden_layer',
                                     activation=tf.nn.relu)
    drop_layer_3 = tf.layers.dropout(inputs=hidden_layer_3,
                                     rate=0.5,
                                     name='third_dropout_layer')

    hidden_layer_4 = tf.layers.dense(inputs=drop_layer_3,
                                     units=n_hidden_4,
                                     name='four_hidden_layer',
                                     activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=hidden_layer_4,
                             units=n_classes,
                             name='outputs')

# Define the loss function for training as cross entropy
with tf.name_scope('loss'):
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xent, name='loss')

# Define the optimiser. Adagrad seems to get the best performance
with tf.name_scope('train'):
    optimiser = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    train_step = optimiser.minimize(loss)

# Output the class probabilities to I can get the AUC
with tf.name_scope('eval'):
    predict = tf.argmax(logits, axis=1, name='class_predictions')
    predict_proba = tf.nn.softmax(logits, name='probability_predictions')

# Initialisation node and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_auc, valid_auc = [], []
n_rounds_not_improved = 0
early_stopping_epochs = 2
with tf.Session() as sess:
    init.run()

    # Begin epoch loop
    print('Training for {} iterations over {} epochs with batchsize {} ...'
          .format(n_iterations, n_epochs, batch_size))
    for epoch in range(n_epochs):

        # Iteration loop
        for iteration in range(n_iterations):
            # Get random selection of data for batch GD. Upsample positive classes to make it
            # balanced in the training batch
            pos_ratio = 0.5
            pos_idx = np.random.choice(np.where(y_train[:, 1] == 1)[0],
                                       size=int(np.round(batch_size * pos_ratio)))
            neg_idx = np.random.choice(np.where(y_train[:, 1] == 0)[0],
                                       size=int(np.round(batch_size * (1 - pos_ratio))))
            idx = np.concatenate([pos_idx, neg_idx])
            batch_X = X_train[idx, :]
            batch_y = y_train[idx, :]

            # Run training
            sess.run(train_step, feed_dict={X: batch_X, y: batch_y})

        # Check on the AUC
        y_pred_train, y_prob_train = sess.run([predict, predict_proba],
                                              feed_dict={X: X_train})
        train_auc.append(roc_auc_score(y_train[:, 1], y_prob_train[:, 1]))

        y_pred_val, y_prob_val = sess.run([predict, predict_proba],
                                          feed_dict={X: X_valid})
        valid_auc.append(roc_auc_score(y_valid[:, 1], y_prob_val[:, 1]))

        # 为了防止过拟合，当模型逐渐变差的时候提前终止Early stopping
        if epoch > 1:
            best_epoch_so_far = np.argmax(valid_auc[:-1])
            if valid_auc[epoch] <= valid_auc[best_epoch_so_far]:
                n_rounds_not_improved += 1
            else:
                n_rounds_not_improved = 0
            if n_rounds_not_improved > early_stopping_epochs:
                print('Early stopping due to no improvement after {} epochs.'
                      .format(early_stopping_epochs))
                break
        print('Epoch = {}, Train AUC = {:.8f}, Valid AUC = {:.8f}'
              .format(epoch, train_auc[epoch], valid_auc[epoch]))

    # Once trained, make predictions
    print('Training complete.')
    y_prob = sess.run(predict_proba, feed_dict={X: predict_df})




fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])
ax.plot(np.arange(len(train_auc)), train_auc, label='Train')
ax.plot(np.arange(len(valid_auc)), valid_auc, label='Valid')
ax.set_xlabel('Epoch')
ax.set_ylabel('AUC')
ax.set_title('Training performance')

fpr, tpr, _ = roc_curve(y_valid[:, 1], y_prob_val[:, 1])
ax1.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(valid_auc[epoch]))
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve')

for a in [ax, ax1]:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.legend(frameon=False)

plt.show()




fig, (ax, ax1) = plt.subplots(1, 2, figsize=[14, 5])

# Precision recall curve
precision, recall, _ = precision_recall_curve(y_valid[:, 1], y_prob_val[:, 1])
ax.step(recall, precision, color='b', alpha=0.2, where='post')
ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Precision - recall curve')

# Confusion matrix
cnf_matrix = confusion_matrix(y_valid[:, 1], np.argmax(y_prob_val, axis=1))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
heatmap = sns.heatmap(cnf_matrix, annot=True, fmt='d', ax=ax1, cmap=cmap, center=0)
ax1.set_title('Confusion matrix heatmap')
ax1.set_ylabel('True label')
ax1.set_xlabel('Predicted label')

plt.show()


out_df = pd.DataFrame({'SK_ID_CURR': meta_df['SK_ID_CURR'][len_train:], 'TARGET': y_prob[:, 1]})
out_df.to_csv('nn_submission.csv', index=False)
