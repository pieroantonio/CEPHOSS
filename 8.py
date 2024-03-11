import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars, OrthogonalMatchingPursuit, RidgeCV, LassoCV, ElasticNetCV, PassiveAggressiveRegressor, RANSACRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump


def load_data_from_excel(filename):
    return pd.read_excel(filename, engine='openpyxl', header=0)


def prepare_data(df, input_features, target):
    df_cleaned = df.dropna(subset=input_features + [target])
    X = df_cleaned[input_features]
    Y = df_cleaned[target]
    return X, Y


def train_and_select_best_model(X_train, y_train, X_test, y_test, model, param_grid):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return best_model, mse, best_params


def main():
    filename = "dati.xlsx"
    df = load_data_from_excel(filename)

    input_features = ['Sex', 'Age', 'WITSA', 'OB_FH', 'OJ_OC', 'U6L6D', 'LFHNP', 'IIANG', 'IMPA', 'COPAD', 'COPOD']
    output_features = ['SNA', 'SNB', 'ANB', 'NP2PA', 'NP2PO', 'SNDST', 'SNFHA', 'CONVX', 'AB2FH', 'TFHNP', 'PLHNP',
                       'U1SNA', 'FMPA', 'SADLA']

    for target in output_features:
        print(f"Training for {target}...")
        X, Y = prepare_data(df, input_features, target)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        models = [
            ('HistGradientBoostingRegressor', HistGradientBoostingRegressor(random_state=42),
             {'model__learning_rate': [0.05, 0.1, 0.2],
              'model__max_iter': [100, 200, 300],
              'model__max_depth': [None, 10, 20]}),
            ('RandomForestRegressor', RandomForestRegressor(random_state=42),
             {'model__n_estimators': [100, 200, 300],
              'model__max_depth': [None, 10, 20],
              'model__min_samples_split': [2, 5, 10]}),
            ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=42),
             {'model__n_estimators': [100, 200, 300],
              'model__max_depth': [None, 10, 20],
              'model__min_samples_split': [2, 5, 10]}),
            ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=42),
             {'model__max_depth': [None, 10, 20],
              'model__min_samples_split': [2, 5, 10]}),
            ('SVR', SVR(),
             {'model__C': [0.1, 1, 10],
              'model__kernel': ['linear', 'rbf', 'poly']}),
            ('LinearSVR', LinearSVR(),
             {'model__C': [0.1, 1, 10],
              'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}),
            ('Ridge', Ridge(),
             {'model__alpha': [0.1, 1, 10],
              'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}),
            ('RidgeCV', RidgeCV(),
             {'model__alphas': [[0.1, 1, 10]]}),
            ('Lasso', Lasso(),
             {'model__alpha': [0.1, 1, 10],
              'model__selection': ['cyclic', 'random']}),
            ('LassoCV', LassoCV(),
             {'model__alphas': [[0.1, 1, 10]]}),
            ('ElasticNet', ElasticNet(),
             {'model__alpha': [0.1, 1, 10],
              'model__l1_ratio': [0.25, 0.5, 0.75],
              'model__selection': ['cyclic', 'random']}),
            ('ElasticNetCV', ElasticNetCV(),
             {'model__alphas': [[0.1, 1, 10]],
              'model__l1_ratio': [0.25, 0.5, 0.75]}),
            ('Lars', Lars(),
             {'model__fit_intercept': [True, False]}),
            ('OrthogonalMatchingPursuit', OrthogonalMatchingPursuit(),
             {'model__n_nonzero_coefs': [None, 5, 10, 20]}),
            ('KNeighborsRegressor', KNeighborsRegressor(),
             {'model__n_neighbors': [3, 5, 10],
              'model__weights': ['uniform', 'distance'],
              'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
            ('XGBRegressor', XGBRegressor(random_state=42),
             {'model__n_estimators': [100, 200, 300],
              'model__learning_rate': [0.01, 0.05, 0.1],
              'model__max_depth': [3, 6, 9]}),
            ('CatBoostRegressor', CatBoostRegressor(random_state=42, verbose=0),
             {'model__n_estimators': [100, 200, 300],
              'model__learning_rate': [0.01, 0.05, 0.1],
              'model__max_depth': [3, 6, 9]}),
            ('MLPRegressor', MLPRegressor(max_iter=2000000, random_state=42),
             {'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
              'model__activation': ['relu', 'tanh'],
              'model__solver': ['adam', 'lbfgs']}),
            ('BaggingRegressor', BaggingRegressor(random_state=42),
             {'model__n_estimators': [10, 50, 100]}),
            ('AdaBoostRegressor', AdaBoostRegressor(random_state=42),
             {'model__n_estimators': [50, 100, 150],
              'model__learning_rate': [0.01, 0.05, 0.1]}),
            ('PassiveAggressiveRegressor', PassiveAggressiveRegressor(random_state=42),
             {'model__C': [0.1, 1, 10],
              'model__fit_intercept': [True, False]}),
            ('RANSACRegressor', RANSACRegressor(random_state=42),
             {'model__min_samples': [None, 0.5, 0.8, 1.0],
              'model__max_trials': [50, 100, 150]}),
            ('BayesianRidge', BayesianRidge(),
             {'model__alpha_1': [1e-06, 1e-05, 1e-04],
              'model__alpha_2': [1e-06, 1e-05, 1e-04],
              'model__lambda_1': [1e-06, 1e-05, 1e-04],
              'model__lambda_2': [1e-06, 1e-05, 1e-04]}),
            ('GaussianProcessRegressor', GaussianProcessRegressor(random_state=42),
             {'model__kernel': [None, 'constant', 'squared_exponential'],
              'model__alpha': [1e-10, 1e-5, 1e-2]}),
            ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=42),
             {'model__n_estimators': [100, 200, 300],
              'model__learning_rate': [0.01, 0.05, 0.1],
              'model__max_depth': [3, 6, 9]})
        ]

        for name, model, param_grid in models:
            print(f"Training for {name}...")
            best_model, best_mse, best_params = train_and_select_best_model(X_train, Y_train, X_test, Y_test, model,
                                                                             param_grid)
            print(f"Best model for {name} with MSE: {best_mse} and best parameters: {best_params}")

            dump(best_model, f'modello_{target}_{name}.joblib')


if __name__ == "__main__":
    main()
