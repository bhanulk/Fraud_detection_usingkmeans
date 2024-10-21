from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import io
import base64

app = Flask(__name__)

# Load dataset
data = pd.read_csv('fraud_0.1origbase.csv')

# Preprocess the data
features = data[['amount', 'oldbalanceOrg', 'newbalanceOrig']]
scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)


def plot_elbow():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def plot_clusters():
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(scaled_features)
    data['Cluster'] = kmeans.labels_

    plt.figure()
    plt.scatter(data['amount'], data['oldbalanceOrg'], c=data['Cluster'], cmap='viridis')
    plt.xlabel('Amount')
    plt.ylabel('Old Balance (Origin)')
    plt.title('K-Means Clustering')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url


def plot_outliers():
    lof = LocalOutlierFactor(n_neighbors=20)
    outliers = lof.fit_predict(scaled_features)
    data['Outlier'] = outliers

    plt.figure()
    plt.scatter(data['amount'], data['oldbalanceOrg'], c=data['Outlier'], cmap='coolwarm')
    plt.xlabel('Amount')
    plt.ylabel('Old Balance (Origin)')
    plt.title('Outliers Detected by LOF')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def home():
    elbow_plot = plot_elbow()
    cluster_plot = plot_clusters()
    outlier_plot = plot_outliers()
    return render_template('index.html', elbow_plot=elbow_plot, cluster_plot=cluster_plot, outlier_plot=outlier_plot)

if __name__ == '__main__':
    app.run(debug=True)
