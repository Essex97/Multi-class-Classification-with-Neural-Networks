import pandas as pd
import numpy as np
import seaborn as sns
import pprint
import codecs
import csv
import time
import math
import datetime
import calendar
import keras.utils
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from imblearn.over_sampling import RandomOverSampler
from ggplot import *
from datetime import date
from matplotlib.pyplot import hist
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from keras import backend as K
from keras import metrics, callbacks, models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adam, Adadelta

DEVELOP = True
TEST = False


# Write csv for the submission
def write_csv(y_pred, name):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Label'])
        for i in range(y_pred.shape[0]):
            writer.writerow([i, y_pred[i]])


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

frame = pd.DataFrame(columns=['Feature', 'MinValue', 'MaxValue'])

for column in df_train:
    if (df_train[column].dtypes == 'float') or (df_train[column].dtypes == 'int') or (column == 'DateOfDeparture'):
        frame = frame.append({
            'Feature': column,
            'MinValue': min(df_train[column]),
            'MaxValue': max(df_train[column])
        }, ignore_index=True)


def split_date(date_time):
    components = date_time.split('-')
    return int(components[0]), int(components[1]), int(components[2])


for dataset in [df_train, df_test]:

    years = []
    months = []
    days = []
    weekDays = []

    for date in dataset['DateOfDeparture']:
        year, month, day = split_date(date)
        years.append(year)
        months.append(month)
        days.append(day)
        date_time = datetime.datetime(year, month, day)
        week_day = calendar.day_name[date_time.weekday()]
        weekDays.append(week_day)

    dataset['Year'], dataset['Month'], dataset['Day'], dataset['WeekDay'] = years, months, days, weekDays

df_train.head()

for dataset in [df_train, df_test]:

    season = []

    for month in dataset['Month']:
        if month in [12, 1, 2]: season.append('WINTER')
        if month in [3, 4, 5]: season.append('SPRING')
        if month in [6, 7, 8]: season.append('SUMMER')
        if month in [9, 10, 11]: season.append('FALL')

    dataset['Season'] = season

df_train.head()

for dataset in [df_train, df_test]:

    vacations = []

    for index, row in dataset.iterrows():
        month, day = row['Month'], row['Day']
        if month == 12 and day in range(10, 31):
            vacations.append('XMAS')
        elif month == 1 and day in range(1, 20):
            vacations.append('NEW_YEAR')
        elif month in [3, 4]:
            vacations.append('SPRING_BREAK')
        elif month == 5 and day in range(25, 31):
            vacations.append('MEMORIAL_DAY')
        elif month == 7 or month == 8:
            vacations.append('SUMMER_VACATIONS')
        elif month == 9 and day in range(1, 7):
            vacations.append('LABOR_DAY')
        elif month == 11 and day in range(22, 28):
            vacations.append('THANKSGIVING')
        else:
            vacations.append('NONE')

    dataset['Vacations'] = vacations


def calculate_distance(LatitudeDeparture, LongitudeDeparture, LatitudeArrival, LongitudeArrival):
    return geopy.distance.distance((LatitudeDeparture, LongitudeDeparture), (LatitudeArrival, LongitudeArrival)).km


for dataset in [df_train, df_test]:

    distances = []

    for index, row in dataset.iterrows():
        distance = calculate_distance(row['LongitudeDeparture'], row['LatitudeDeparture'], row['LongitudeArrival'],
                                      row['LatitudeArrival'])
        distances.append(int(round(distance, 0)))

    dataset['Distance'] = distances


mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x,mlab.normpdf(x, mu, sigma))
plt.axvline(-1, color='r')
plt.axvline(1, color='r')
plt.axvline(0, color='g', linestyle='--')
plt.scatter([-1, 1], [0.25, 0.25])
plt.show()


for dataset in [df_train, df_test]:
    mean_minus_std = []
    mean_plus_std = []

    for index, row in dataset.iterrows():
        mean_minus_std.append(row['WeeksToDeparture'] - row['std_wtd'])
        mean_plus_std.append(row['WeeksToDeparture'] + row['std_wtd'])

    dataset['mean_minus_std'] = mean_minus_std
    dataset['mean_plus_std'] = mean_plus_std

if TEST:
    y_train = df_train[['PAX']]

    df_train.drop(df_train.columns[[0,2,6,9,10,11]], axis=1, inplace=True)

    df_test.drop(df_test.columns[[0,2,6,9,10]], axis=1, inplace=True)

if DEVELOP:
    y_train = df_train[['PAX']]

    df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)

    y_test = df_test[['PAX']]

    df_train.drop(df_train.columns[[0,2,6,9,10,11]], axis=1, inplace=True)

    df_test.drop(df_test.columns[[0,2,6,9,10,11]], axis=1, inplace=True)


def normalize_feature(feature_name, dataset):
    min_dataset = dataset[feature_name].min()
    max_dataset = dataset[feature_name].max()
    new_dataset = []

    for index, row in dataset.iterrows():
        new_dataset.append((row[feature_name] - min_dataset) / (max_dataset - min_dataset))

    dataset[feature_name] = new_dataset

features_to_use = ['Distance', 'LongitudeDeparture', 'LatitudeDeparture', 'LongitudeArrival', 'LatitudeArrival', 'Day', 'mean_minus_std', 'mean_plus_std']

for dataset in [df_train, df_test]:
  for feature in features_to_use:
    normalize_feature(feature, dataset)

df_train.head()


def encode_feature(feature_name):
    encoder = LabelEncoder()
    encoder.fit(df_train[feature_name])

    df_train[feature_name] = encoder.transform(df_train[feature_name])
    df_test[feature_name] = encoder.transform(df_test[feature_name])


columns_to_feature = ['Departure', 'Arrival', 'Year', 'Month', 'WeekDay', 'Season', 'Vacations']

for feature_name in columns_to_feature:
    encode_feature(feature_name)

extra_train_df = pd.concat(
    [df_train['Departure'], df_train['Arrival'], df_train['Year'], df_train['Month'], df_train['WeekDay'],
     df_train['Day'], df_train['Season'], df_train['Vacations']], axis=1)
extra_test_df = pd.concat(
    [df_test['Departure'], df_test['Arrival'], df_test['Year'], df_test['Month'], df_test['WeekDay'], df_test['Day'],
     df_test['Season'], df_test['Vacations']], axis=1)

pca = PCA(n_components=1)
pca.fit(extra_train_df)

X_train_extra = pca.transform(extra_train_df)
df_train['embending'] = X_train_extra
normalize_feature('embending', df_train)

X_test_extra = pca.transform(extra_test_df)
df_test['embending'] = X_test_extra
normalize_feature('embending', df_test)

df_train.head()

pca = PCA(n_components=1)
pca.fit(pd.concat([df_train['LongitudeDeparture'], df_train['LatitudeDeparture'], df_train['Distance']], axis=1))
ldd_train = pca.transform(
    pd.concat([df_train['LongitudeDeparture'], df_train['LatitudeDeparture'], df_train['Distance']], axis=1))
ldd_test = pca.transform(
    pd.concat([df_test['LongitudeDeparture'], df_test['LatitudeDeparture'], df_test['Distance']], axis=1))

df_train['ldd'] = ldd_train
normalize_feature('ldd', df_train)

df_test['ldd'] = ldd_test
normalize_feature('ldd', df_test)

pca = PCA(n_components=1)
pca.fit(pd.concat([df_train['LongitudeArrival'], df_train['LatitudeArrival'], df_train['Distance']], axis=1))
lad_train = pca.transform(
    pd.concat([df_train['LongitudeArrival'], df_train['LatitudeArrival'], df_train['Distance']], axis=1))
lad_test = pca.transform(
    pd.concat([df_test['LongitudeArrival'], df_test['LatitudeArrival'], df_test['Distance']], axis=1))

df_train['lad'] = lad_train
normalize_feature('lad', df_train)

df_test['lad'] = lad_test
normalize_feature('lad', df_test)

corr = df_train.corr()


def one_hot_formulation(feature_name, df_train, df_test):
    new_df_train = pd.concat([df_train, pd.get_dummies(df_train[feature_name], prefix=feature_name)], axis=1)
    # new_df_train.drop([feature_name], axis=1, inplace=True)
    normalize_feature(feature_name, new_df_train)
    new_df_test = pd.concat([df_test, pd.get_dummies(df_test[feature_name], prefix=feature_name)], axis=1)
    # new_df_test.drop([feature_name], axis=1, inplace=True)
    normalize_feature(feature_name, new_df_test)

    return new_df_train, new_df_test


features = ['Departure', 'Arrival', 'Year', 'Month', 'WeekDay', 'Season', 'Vacations']

for feature_name in features:
    df_train, df_test = one_hot_formulation(feature_name, df_train, df_test)
df_train.head()

SHOW_GRAPH = True

if SHOW_GRAPH:
    tsne = TSNE(n_components=2)
    transformed_tsne_data = tsne.fit_transform(df_train)

    tsne_dataframe = pd.DataFrame(transformed_tsne_data)
    pax_dataframe = pd.DataFrame(y_train)

    tsne_dataframe = pd.concat([tsne_dataframe, pax_dataframe], axis=1, ignore_index=True)

    tsne_dataframe.columns = ['X', 'Y', 'label']

    tsne_dataframe['label'] = tsne_dataframe['label'].astype(str)

    chart = ggplot(tsne_dataframe, aes(x='X', y='Y', color='label')) + geom_point(alpha=0.8) + ggtitle("tSNE reduction")
    chart.show()

sns.heatmap(corr,
            cmap="RdBu_r",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

print(pd.value_counts(y_train.values.flatten()).plot(kind='bar'))


# --------------------------------------Classifiers---------------------------------
EXEC = False

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

if EXEC:
    start = time.time()

    # -- Logistic Regression -- #
    clf = LogisticRegression()
    # clf.fit(X_train_2_fit, y_train)

    # -- Linear Regression -- #
    clf1 = LinearRegression()
    # clf1.fit(X_train_2_fit, y_train)

    clf2 = GradientBoostingClassifier(n_estimators=600, max_depth=None, learning_rate=0.1)
    # clf2.fit(X_train_2_fit, y_train)

    # -- Desicion Tree --#
    clf3 = DecisionTreeClassifier()
    # clf3.fit(X_train_2_fit, y_train)

    # -- Knn Classifier --#
    clf4 = KNeighborsClassifier(n_neighbors=15, algorithm='ball_tree')
    clf4.fit(X_train_2_fit, y_train)

    estimators = []
    estimators.append(('logistic', clf))
    # estimators.append(('linear', clf1))
    estimators.append(('DSC', clf3))
    # estimators.append(('KNN', clf4))

    # ensemble = VotingClassifier(estimators, weights=[1,2])
    # ensemble.fit(X_train_2_fit, y_train)

    # y_pred = ensemble.predict(X_test_2_fit)

    y_pred = clf4.predict(X_test_2_fit)

    if DEVELOP:
        print(f1_score(y_test, y_pred, average='micro'))
    elif TEST:
        write_csv(y_pred=y_pred)

    end = time.time()
    print('time: {0:f}'.format(end - start))

# ----------------------------------------------------------------------


def f1_score_metric(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?features_to_use = ['Distance', 'LongitudeDeparture', 'LatitudeDeparture', 'LongitudeArrival', 'LatitudeArrival', 'mean_minus_std', 'mean_plus_std', 'ticket']

    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


# --------------------------------------Neural Network---------------------------------


def voting(y1, y2, y3):
    y_best = y1.copy()

    for i in range(len(y1)):
        if y2[i] == y3[i]:
            y_best[i] = y2[i]

    return y_best


EXEC = True

if EXEC:
    start = time.time()

    X_train = df_train
    X_test = df_test
    y_train = np.ravel(y_train)
    new_y_pred = []  # The list of the predictions from each iteration

    sampler = RandomOverSampler('minority')
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    INPUT_LEN = X_train.shape[1]
    one_hot_labels_train = pd.get_dummies(y_train)

    for i in [1, 2, 3]:
        model1 = Sequential([
            Dense(128, input_shape=(INPUT_LEN,), kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dropout(0.3),
            Dense(64, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(16, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dense(8, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('softmax')
        ])

        model2 = Sequential([
            Dense(512, input_shape=(INPUT_LEN,), kernel_initializer='he_uniform', bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0.3),
            Dense(256, kernel_initializer='he_uniform', bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(128, kernel_initializer='he_uniform', bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(32, kernel_initializer='he_uniform', bias_initializer='Ones'),
            Activation('relu'),
            Dense(8),
            Activation('softmax')
        ])

        model3 = Sequential([
            Dense(256, input_shape=(INPUT_LEN,), kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dropout(0.2),
            Dense(128, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(64, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(16, kernel_initializer='he_uniform', bias_initializer='zeros'),
            Activation('relu'),
            Dense(8),
            Activation('softmax')
        ])

        model4 = Sequential([
            Dense(128, input_shape=(INPUT_LEN,), bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0.2),
            Dense(64, bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(32, bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 1),
            Dense(16, bias_initializer='Ones'),
            Activation('relu'),
            Dense(8),
            Activation('softmax')
        ])

        model5 = Sequential([
            Dense(256, input_shape=(INPUT_LEN,), bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0.2),
            Dense(128, bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(64, bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 2),
            Dense(32, bias_initializer='Ones'),
            Activation('relu'),
            Dropout(0, 1),
            Dense(16, bias_initializer='Ones'),
            Activation('relu'),
            Dense(8),
            Activation('softmax')
        ])

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        addelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        model1.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', f1_score_metric])
        model1.fit(X_train, one_hot_labels_train, shuffle=True, epochs=(75))
        y_pred1 = model1.predict(X_test)

        model2.compile(optimizer=addelta, loss='categorical_crossentropy', metrics=['accuracy', f1_score_metric])
        model2.fit(X_train, one_hot_labels_train, epochs=(28))
        y_pred2 = model2.predict(X_test)

        model3.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', f1_score_metric])
        model3.fit(X_train, one_hot_labels_train, shuffle=True, epochs=(20))
        y_pred3 = model3.predict(X_test)

        model4.compile(optimizer=addelta, loss='categorical_crossentropy', metrics=['accuracy', f1_score_metric])
        model4.fit(X_train, one_hot_labels_train, shuffle=True, epochs=(80))
        y_pred4 = model4.predict(X_test)

        model5.compile(optimizer=addelta, loss='categorical_crossentropy', metrics=['accuracy', f1_score_metric])
        model5.fit(X_train, one_hot_labels_train, epochs=(38))
        y_pred5 = model5.predict(X_test)

        y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5  # Compute the average of each model probalities
        y_pred = y_pred.argmax(axis=1)  # Translate the results into categories
        new_y_pred.append(y_pred)

    y_pred = voting(new_y_pred[0], new_y_pred[1], new_y_pred[2])

    if TEST:
        write_csv(y_pred, 'y_pred_best.csv')
    elif DEVELOP:
        print(f1_score(y_pred, y_test, average='micro'))

    end = time.time()
    print('time: {0:f}'.format(end - start))

