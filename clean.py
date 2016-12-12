import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def load_data():
    df = pd.read_csv('pumpituptrain.csv')
    y = pd.read_csv('pumpituplables.csv')
    return df, y

def plot_eda(df, y):
    fig = plt.figure(fig_size=(10,10))
    ax = fig.add_subplot(1,1,1)
    df_all = pd.merge(df, y, on = 'id')

def predict_population(df):
    train = df[df['population']!=0]
    for region in train.region.unique():
        train = df[df['population']!=0]
        train = train[train['region']==region]
        x = train[['latitude', 'longitude']].values
        y = train['population'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
        knn.fit(x_train, y_train)

def predict_construction_year(df):
    train = df[df['construction_year']!=0]
    y = train.pop('construction_year')
    train[['extraction_type_class', 'install_by_dwe', 'install_by_wv', 'install_by_commu', 'install_by_hesawa']]
    dummies=pd.get_dummies(train['extraction_type_class'], drop_first=True)
    train = pd.concat([train, dummies], axis=1)
    x = train.values
    rf = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    print cross_val_score(rf, x, y)


def clean(df):
    df = df[['amount_tsh', 'gps_height', 'installer', 'longitude', 'latitude', 'basin', 'population', 'extraction_type_class', 'payment', 'construction_year', 'quantity', 'region']]
    df['install_by_dwe'] = df['installer']=='DWE'
    df['install_by_commu'] = df['installer']=='Commu'
    df['install_by_hesawa'] = df['installer']=='Hesawa'
    df['install_by_wv'] = df['installer']=='World Vision'
    df['intall_by_other']= [1 if row not in ('DWE','Commu', 'Hesawa', 'World Vision') else 0 for row in df['installer']]
    pd.get_dummies(df['basin'], drop_first=True)
    pd.get_dummies(df['extraction_type_class'], drop_first=True)
    #pd.get_dummies(df['payment'], drop_first=True)
    #pd.get_dummies(df['quantity'], drop_first=True)
    df['longitude'].replace(0, np.NaN)
    df["longitude"] = df.groupby("region").transform(lambda x: x.fillna(x.mean()))
    df['population'].replace(0, np.NaN, inplace=True)
    df['population'] = df.groupby('region').transform(lambda x: x.fillna(x.mean()))
    df= df.drop('region', 1)
    #df = df.drop(['id', 'scheme_name', 'installer', 'extraction_type', 'extraction_type_group', 'funder', 'wpt_name', 'subvillage', 'region', 'region_code', 'distric_code', 'lga', 'ward', ''], 1)
    return df

def main():
    df, y = load_data()
    df = clean(df)
    #predict_population(df)
    predict_construction_year(df)
    return df, y





if __name__ == '__main__':
    df, y = main()
