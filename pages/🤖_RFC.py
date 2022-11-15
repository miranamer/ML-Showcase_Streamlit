import streamlit as st
from sklearn.ensemble import RandomForestClassifier
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

st.markdown("<h1 style='text-align: center; color: white;'>Random Forest Classifier</h1>", unsafe_allow_html=True)

st.image("assets/random_forest.png", caption="RFC Visualised")

st.write("Random forest is a Supervised Machine Learning Algorithm that is used widely in Classification and Regression problems. It builds decision trees on different samples and takes their majority vote for classification and average in case of regression.")
st.markdown("<p style='text-align: center; text-decoration: underline; color: white;'>Gini Impurity Formula</p>", unsafe_allow_html=True)
st.latex(r"\sum_{i=1}^C f_i\left(1-f_i\right)")
st.write("Gini Impurity is a measurement used to build Decision Trees to determine how the features of a dataset should split nodes to form the tree.")

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

st.markdown("""---""")

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

st.markdown("""---""")

st.subheader("PCA Graph")
st.pyplot(fig)

num_classes = len(np.unique(y))

st.markdown("""---""")

st.subheader("Customise The Algorithm Parameters")

n_estimators = st.slider('N Estimators', 1, 200)
criterion = st.selectbox("Choose A Criterion For RFC", ("Gini", "Entropy", "Log Loss"))


criterion_map = {"Gini": 'gini', "Entropy": 'entropy', "Log Loss": 'log_loss'}

clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion_map[criterion])

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
