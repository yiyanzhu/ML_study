import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/train_pre.csv', header=0,  encoding='GBK')
df_ex = pd.read_csv('data/test.csv', header=0,  encoding='GBK')
"""y_train = df.iloc[:, [0, 1]]
y_train.set_index('PassengerId', inplace=True)
y_train = np.ravel(y_train.values)
y_train = np.ravel(df.iloc[:, [0, 1]].set_index('PassengerId').values)
print(y_train)"""
y = np.ravel(df.iloc[:, [1]].values) #.values返还为numpy数组，会舍去dataframe的索引，np.ravel将数组展平成一维数组。
X_pre = df.drop(columns=['Survived', 'Ticket', '登船地点'])
X = pd.concat([X_pre, df_ex], axis=0, ignore_index=True) #ignore_index=True
"""没有利用算法如主成分分析，互信息等进行特征选择
利用随机森林模型的feature_importances_，SHAP、LIME进行最终模型的特征分析
扔掉姓名，船票名，登船地点。阶级，兄弟姐妹数量,父母或者子女数量是有序离散变量。性别是无序离散变量。Age,船票价格是连续变量。共六个特征用于训练"""
#problem1_使用多个独热编码会出现空间太大，常常和PCA一起使用。problem2_年龄有大量缺失值，用的均值进行填充

#填充缺失值
mean_value_age = X['Age'].mean()
X['Age'].fillna(mean_value_age, inplace=True)
mean_value_price = X['船票价格'].mean()
X['船票价格'].fillna(mean_value_price, inplace=True)
nan = X.isnull().sum()

#性别独热编码
encoded_df = pd.get_dummies(X['Sex'])
X.drop(columns=['Sex'], inplace=True)
X = pd.concat([X, encoded_df], axis=1)

#标准化
selected_columns = X.iloc[:, 2:6]
index_column = X.iloc[:, 0]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_columns)
scaled_df = pd.DataFrame(scaled_data, columns=selected_columns.columns)
sex = X.iloc[:, 6:8]
X = pd.concat([scaled_df, sex], axis=1)
X = pd.concat([index_column, X], axis=1)
X.set_index('PassengerId', inplace=True)
Target = X.iloc[891:, :]
X = X.iloc[:891, :]

#pipeline+gridsearch. 将数据集划分为训练集和测试集，对训练集的每个超参进行k折交叉验证，每一折的验证数据集进行打分取平均值，相互独立，训练参数不会继承
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

"""#支持向量机gs
from sklearn.svm import SVC
pipe_svc = make_pipeline(SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
#param_grid = [{'svc__C': [1.0], 'svc__gamma': [0.1], 'svc__kernel': ['rbf']}]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
clf = gs.best_estimator_
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}') #:.3f保留三位浮点数作为输出"""
"""0.825802034428795 {'svc__C': 1.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'} Test accuracy: 0.810"""


#多层感知器gs
"""from sklearn.neural_network import MLPClassifier
param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(50, 50)],
    'mlpclassifier__activation': ['relu'],
    'mlpclassifier__solver': ['adam'],
    'mlpclassifier__alpha': [0.01],
    'mlpclassifier__max_iter': [500, 600, 700]
}
pipe_mlp = make_pipeline(MLPClassifier(random_state=1))
gs = GridSearchCV(estimator=pipe_mlp, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print("Best score:", gs.best_score_, "Best parameters:", gs.best_params_)
clf = gs.best_estimator_
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')"""

#随机森林gs
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300, 400],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 3]
}
pipe_rf = make_pipeline(RandomForestClassifier(random_state=1))
gs = GridSearchCV(estimator=pipe_rf, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs.fit(X_train, y_train)
print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)
clf = gs.best_estimator_
print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')

rf_model = clf.named_steps['randomforestclassifier']
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

"""#随机森林效果最好，用来预测新数据
best_model = gs.best_estimator_
predictions = best_model.predict(Target)
predictions = pd.DataFrame(predictions)
index = df_ex.iloc[:, 0]
outcome = pd.concat([index, predictions], axis=1)
outcome.to_csv('Titanic_predictions.csv', index=False)"""

