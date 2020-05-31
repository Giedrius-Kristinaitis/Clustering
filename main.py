from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


def plot_clusters(column_data, columns, km_result, centroid_count) -> None:
    cluster_colors = ['green', 'red', 'blue', 'yellow']

    for i in range(centroid_count):
        plt.scatter(
            column_data[km_result == i, 0],
            column_data[km_result == i, 1],
            s=50,
            c=cluster_colors[i],
            marker='s',
            edgecolor='black',
            label=f"cluster {i + 1}"
        )

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.show()

def execute(data, columns, centroid_count) -> None:
    column_data = data[columns].to_numpy()

    km = KMeans(n_clusters=centroid_count)
    km_result = km.fit_predict(column_data)

    plot_clusters(column_data, columns, km_result, centroid_count)

data = pd.read_csv('cpu.csv')

# classify data into 2 clusters
execute(data, ['cach', 'perf'], 2)
execute(data, ['mmin', 'estperf'], 2)
execute(data, ['chmax', 'perf'], 2)

# classify data into 3 clusters
execute(data, ['cach', 'perf'], 3)
execute(data, ['mmin', 'estperf'], 3)
execute(data, ['chmax', 'perf'], 3)

# classify data into 4 clusters
execute(data, ['cach', 'perf'], 4)
execute(data, ['mmin', 'estperf'], 4)
execute(data, ['chmax', 'perf'], 4)