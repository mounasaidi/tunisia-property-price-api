from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def train_models(X_train, y_train):
    models = {
        "lr": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=200),
        "gb": GradientBoostingRegressor(n_estimators=200),
        "xgb": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8
        )
    }

    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    
    print(X_train.columns)
    return trained