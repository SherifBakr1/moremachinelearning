import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.cluster import DBSCAN
import numpy as np

df = pd.read_csv('C:\\Users\\Sheri\\Documents\\COMP4432\\Assignment 5\\network.csv')

df['day'] = pd.to_datetime(df['day'])

df_last_day = df[df['day'] == '2021-06-13'].drop(columns=['day'])

bts_id = df_last_day['BTS_ID']
df_last_day = df_last_day.drop(columns=['BTS_ID'])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_last_day)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

pca_df['BTS_ID'] = bts_id.values

fig = px.scatter(pca_df, x='PC1', y='PC2', hover_data=['BTS_ID'])
fig.show()

dbscan = DBSCAN(eps=1, min_samples=2)
pca_df['labels'] = dbscan.fit_predict(principal_components)

outliers_bts_id = pca_df[pca_df['labels'] == -1]['BTS_ID']
print("BTS_IDs of outliers:", outliers_bts_id.tolist())

pca_df['labels'] = pca_df['labels'].apply(lambda x: -1 if x == -1 else 0)

fig = px.scatter(pca_df, x='PC1', y='PC2', color='labels', hover_data=['BTS_ID'], color_continuous_scale=px.colors.sequential.Viridis)
fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig.show()

df['day'] = pd.to_datetime(df['day'])

df_excluded = df[df['day'] != '2021-06-13'].copy()
df_included = df[df['day'] == '2021-06-13'].copy()

def preprocess_and_pca(df):
    df_temp = df.drop(columns=['day', 'BTS_ID'])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_temp)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    return principal_components

principal_components_excluded = preprocess_and_pca(df_excluded)
df_excluded['PC1'], df_excluded['PC2'] = principal_components_excluded[:,0], principal_components_excluded[:,1]

df_excluded['distance'] = np.sqrt(df_excluded['PC1']**2 + df_excluded['PC2']**2)
mean_distances = df_excluded.groupby('BTS_ID')['distance'].mean()

principal_components_included = preprocess_and_pca(df_included)
df_included['PC1'], df_included['PC2'] = principal_components_included[:,0], principal_components_included[:,1]
df_included['distance'] = np.sqrt(df_included['PC1']**2 + df_included['PC2']**2)

dbscan_included = DBSCAN(eps=1, min_samples=2)
df_included['labels'] = dbscan_included.fit_predict(principal_components_included)

df_included['mean_distance'] = df_included['BTS_ID'].map(mean_distances)
df_included['improvement'] = df_included['distance'] < df_included['mean_distance']

print("Sites getting worse (outliers and further from the center):")
print(df_included[(df_included['labels'] == -1) & (df_included['improvement'] == False)]['BTS_ID'].tolist())

print("Sites getting better:")
print(df_included[df_included['improvement'] == True]['BTS_ID'].tolist())

# Conclusion about engineers' progress
if len(df_included[df_included['improvement'] == True]) > len(df_included[(df_included['labels'] == -1) & (df_included['improvement'] == False)]):
    print("Engineers are making progress with system performance.")
else:
    print("More work is needed to improve system performance.")

