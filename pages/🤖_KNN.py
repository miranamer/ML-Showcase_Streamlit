import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.offline as py
import matplotlib.pyplot as plt

st.sidebar.success("Select A Page")

st.markdown("<h1 style='text-align: center; color: white;'>K Nearest Neighbour</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')

with col2:
    st.image("assets/knn_logo.png", caption="KNN Visualised")

with col3:
    st.write(' ')

st.write("The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.")

st.markdown("<p style='text-align: center; text-decoration: underline; color: white;'>Minkowski Distance Formula</p>", unsafe_allow_html=True)
st.latex(r"\operatorname{dist}(\mathbf{x}, \mathbf{z})=\left(\sum_{r=1}^d\left|x_r-z_r\right|^p\right)^{1 / p}")
st.write("The k-nearest neighbor classifier fundamentally relies on a distance metric. The better that metric reflects label similarity, the better the classified will be. The most common choice is the Minkowski distance")

st.markdown("""---""")

possible_datasets = ("Iris", "Breast Cancer", "Wine")

st.subheader("Datasets:")
chosen_dataset = st.selectbox("Choose A Dataset", possible_datasets)


@st.cache
def retrieve_data():
    data_dict = {"Iris": load_iris(), "Breast Cancer": load_breast_cancer(), "Wine": load_wine()}
    data = data_dict[chosen_dataset]
    return data


load_text = st.text("Loading Data...")
data = retrieve_data()
load_text.text("Data Loaded!")

X, y = data.data, data.target

df = pd.DataFrame(data=X, columns=data.feature_names)
df["target"] = data.target

st.subheader("Raw Data:")
st.write(df.head())

st.subheader("Data For First 30 Records:")
st.bar_chart(df.head(30))

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()

st.subheader("PCA Graph")
st.pyplot(fig)

num_classes = len(np.unique(y))

n_neighbours = st.slider('K', 1, 15)
leaf_size = st.slider('Leaf Size', 1, 30)
algorithm = st.selectbox("Choose An Algorithm For KNN", ("Ball Tree", "KD Tree", "Brute", "Auto"))

algorithm_map = {"Ball Tree": 'ball_tree', "KD Tree": 'kd_tree', "Brute": 'brute', "Auto": 'auto'}

clf = KNeighborsClassifier(n_neighbors=n_neighbours, leaf_size=leaf_size, algorithm=algorithm_map[algorithm])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.markdown("""---""")

st.subheader("Confusion Matrix")
st.write("Values starting from the top left diagonal are correct predictions, all else are mismatches the model had.")

matrix = plot_confusion_matrix(clf, X_test, y_test)
matrix.ax_.set_title("Confusion Matrix", color='black')
plt.xlabel("Predicted Label", color='black')
plt.ylabel("True Label", color='black')
plt.gcf().axes[0].tick_params(colors='black')
plt.gcf().axes[1].tick_params(colors='black')
plt.gcf().set_size_inches(10, 6)
st.pyplot(plt)

st.markdown("""---""")

st.subheader("Final Accuracy")
formatted_num = '{0:.2f}'.format(accuracy_score(y_test, y_pred) * 100)
accuracy = st.success(f"Accuracy: {formatted_num}%")




