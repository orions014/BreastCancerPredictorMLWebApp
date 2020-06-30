import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class CancerClassifier:

    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.tree_classifier = self.classifier()

    def predict(self, feature_value):
        sc = StandardScaler()
        my_prediction = self.tree_classifier.predict(sc.fit_transform([feature_value]))
        return my_prediction

    def classifier(self):

        df_breast_cancer = pd.read_csv('./breastcancerdetector/newbreast_cancer.csv')

        # Data preprocessing
        # removing non numeric value
        data_columns = list(df_breast_cancer.columns)
        df_breast_cancer = (df_breast_cancer.drop(data_columns, axis=1)
                     .join(df_breast_cancer[data_columns].apply(pd.to_numeric, errors='coerce')))

        df_breast_cancer = df_breast_cancer[df_breast_cancer[data_columns].notnull().all(axis=1)]

        input_features = df_breast_cancer.iloc[:, 1:10].values
        output_value = df_breast_cancer.iloc[:, 10].values

        X_train, X_test, y_train, y_test = train_test_split(input_features, output_value, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        classifier_model = self.get_classifier(X_train,y_train,X_test,y_test)
        return classifier_model

    def get_classifier(self, X_train, y_train,X_test,y_test):
        maximum_accuracy = 0
        model = 0
        # Logistic regression
        #model 0
        logistict_regression = LogisticRegression(random_state=0)
        logistict_regression.fit(X_train, y_train)
        log_accuracy = logistict_regression.score(X_test,y_test)


        # Decision Tree
        #model 1
        tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        tree.fit(X_train, y_train)
        tree_accuracy = tree.score(X_test,y_test)


        # Random Forest Classifier
        #model 2
        forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        forest.fit(X_train, y_train)
        forest_accuracy = forest.score(X_test,y_test)

        model_array = [logistict_regression,tree,forest]
        if maximum_accuracy < log_accuracy:
            maximum_accuracy = log_accuracy
            model = 0
        if maximum_accuracy < tree_accuracy:
            maximum_accuracy = tree_accuracy
            model = 1
        if maximum_accuracy < forest_accuracy:
            maximum_accuracy = forest_accuracy
            model = 2


        return model_array[model]
