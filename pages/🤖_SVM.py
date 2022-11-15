from sklearn.svm import SVC
import streamlit as st
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
from sklearn import metrics


st.sidebar.success("Select A Page")

st.markdown("<h1 style='text-align: center; color: white;'>Support Vector Machine</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("assets/SVM_2.png", caption="SVM Visualised")

with col3:
    st.write(' ')


st.write("A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they're able to categorize new text.")

st.write("Compared to newer algorithms like neural networks, they have two main advantages: higher speed and better performance with a limited number of samples (in the thousands). This makes the algorithm very suitable for text classification problems, where its common to have access to a dataset of at most a couple of thousands of tagged samples.")
st.markdown("<p style='text-align: center; text-decoration: underline; color: white;'>RBF Kernel Formula</p>", unsafe_allow_html=True)
st.latex(r"f(x 1, x 2)=e^{\frac{-\|(x 1-x 2)\|^2}{2 \sigma^2}}")
st.write("What this formula actually does is create non-linear combinations of our features to lift your samples onto a higher-dimensional feature space where we can use a linear decision boundary to separate your classes It is the most used kernel in SVM classifications, the following formula explains it mathematically")

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

C = st.slider('C', 1, 100)
kernel = st.selectbox("Choose A kernel For SVM", ("Linear", "Poly", "Rbf", "Sigmoid"))

kernel_map = {"Linear": "linear", "Poly": "poly", "Rbf": "rbf", "Sigmoid": "sigmoid"}

clf = SVC(C=C, kernel=kernel_map[kernel])

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
