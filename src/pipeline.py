from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import time

from config import CLASSIFIER
from data import get_training_data
from metrics import evaluate_pipeline


def get_pipeline(df):
    categorical_features = df.select_dtypes(include=['object', 'int64']).columns
    numerical_features = df.select_dtypes(include=['float64']).columns

    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('classifier', CLASSIFIER)
    ])
    
    return pipeline

def run_experiment(save_result=False):
    warnings.simplefilter(action='ignore', category=FutureWarning)

    start_time = time.time()

    X_train, X_val, y_train, y_val = get_training_data()

    pipeline = get_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    end_time = time.time()

    execution_time = end_time - start_time

    evaluate_pipeline(pipeline, X_train, X_val, y_train, y_val, execution_time, save_result)
