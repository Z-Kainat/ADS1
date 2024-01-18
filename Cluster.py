import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np
import scipy.optimize as opt

# Create a StandardScaler instance
scaler = StandardScaler()

def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhouette score
    score = skmet.silhouette_score(xy, labels)
    return score


def plot_clusters(df_cluster, x_label, y_label, output_filename, n_clusters=2):
    """Fit KMeans clustering and plot the clusters"""
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(df_cluster)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    plt.figure(figsize=(6.0, 6.0))
    # scatter plot with colours selected using the cluster numbers
    plt.scatter(df_cluster[x_label], df_cluster[y_label], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    xc, yc = cen[:, 0], cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    # c = colour, s = size
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')
    plt.title(f"{x_label} & {y_label}", fontweight='bold')
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Example usage
file_paths = [
    'Agricultural land.csv',
    "Access to electricity.csv",
    'Electricity coal sources.csv',
    'CO2 emissions.csv',
    'greenhouse gas.csv'
]
selected_country = "Pakistan"
start_year = 1990
end_year = 2022

result_df = read_data(file_paths, selected_country, start_year, end_year)
# Remove the 'Year' column
result_df = result_df.drop('Year', axis=1)

# Apply Standard Scaling to all columns in the DataFrame
result_df_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns)

print(result_df_scaled)

# scatter plot
pd.plotting.scatter_matrix(result_df_scaled, figsize=(9.0, 9.0))
plt.tight_layout()  # helps to avoid overlap of labels
plt.show()

# Calculate the correlation matrix
correlation_matrix = result_df_scaled.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix between indicators that affect climate change', fontweight='bold')
# Rotate x-axis labels to 45 degrees
plt.yticks(rotation=0,fontweight='bold')
plt.xticks(rotation=0,fontweight='bold')
plt.savefig('Correlation', dpi=300)
plt.show()

# Cluster: Electricity coal sources & CO2 emissions
df_cluster1 = result_df[['Electricity coal sources', 'CO2 emissions']]
plot_clusters(df_cluster1, 'Electricity coal sources', 'CO2 emissions', 'cluster_Electricity_Coal_CO2', n_clusters=3)

# Cluster: Access to electricity & CO2 emissions
df_cluster2 = result_df[['Access to electricity', 'CO2 emissions']]
plot_clusters(df_cluster2, 'Access to electricity', 'CO2 emissions', 'cluster_Electricity_CO2', n_clusters=2)
