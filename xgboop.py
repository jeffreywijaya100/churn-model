import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Data_C_Model:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file, delimiter=',')
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def preprocess_data(self):
        self.df = self.df.drop(['Unnamed: 0','id','CustomerId','Surname'], axis = 1)
        
        # Replace missing values
        self._replace_missing_values(self.df)
        self._replace_missing_values(self.df)

        # Encode categorical features
        self._encode_categorical_features(self.df)
        self._encode_categorical_features(self.df)
        
        input_df = self.df.drop('churn', axis=1)
        output_df = self.df['churn']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input_df, output_df, test_size=0.2, random_state=42)

    def _replace_missing_values(self, data):
        data['CreditScore'].fillna(round(data['CreditScore'].mean(),0), inplace=True)

    def _encode_categorical_features(self, data):
        encode_mapping = {"Gender": {"Male": 1, "Female": 0}, 'Geography': {'Germany':3,'Spain':2,'France':1}}
        data.replace(encode_mapping, inplace=True)

    def train_model(self):
        xgb_class = xgb.XGBClassifier()
        xgb_class.fit(self.x_train, self.y_train)
        return xgb_class

    def evaluate_model(self, model):
        y_predict = model.predict(self.x_test)
        print('\nClassification Report\n')
        print(classification_report(self.y_test, y_predict, target_names=['0', '1']))

def main():
    model = Data_C_Model('data_C.csv')
    model.preprocess_data()
    trained_model = model.train_model()
    model.evaluate_model(trained_model)

if __name__ == "__main__":
    main()
