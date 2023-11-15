import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Model:
    def __init__(self):
        self.X_train = None
        self.features = None
        self.model = self.train()

    def train(self):
        data = pd.read_csv("diabetes_prediction_dataset.csv")

        label_encoders = {}
        categorical_columns = ['gender', 'smoking_history']
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        data = pd.concat([data, data.iloc[:, -1]], axis=1)

        X = data.drop("diabetes", axis=1).to_numpy()
        y = data["diabetes"].to_numpy()
        y = [1 - value for value in y]

        self.features = data.columns.drop("diabetes")

        self.X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(hidden_layer_sizes=(5,),activation='logistic', max_iter=10000,learning_rate='invscaling',random_state=0)
        model = make_pipeline(StandardScaler(), mlp)

        model.fit(self.X_train, y_train)
        return model

    def predict(self, data):
        data = data.reshape(1, -1)

        pred = self.model.predict(data)
        conf = self.model.predict_proba(data)[0][pred]

        plt.clf()

        explainer = shap.KernelExplainer(self.model.predict, self.X_train[:100])
        shap_values = explainer.shap_values(data, nsamples=100)
        shap.force_plot(explainer.expected_value, shap_values[0], self.X_train[0,:],feature_names=self.features, matplotlib=True, show=False)

        plt.savefig('temp.png')
        plt.close()

        return pred[0], conf[0]