from skopt import gp_minimize
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

def tune_xgb(
    X_train, y_train, X_valid, y_valid,
    num_classes=2,
    n_calls=30,
    random_state=42,
):
    """
    Tune hyperparameters for XGBoost using Bayesian optimization.
    Parameters:
        X_train: Training feature set.
        y_train: Training labels.
        X_valid: Validation feature set.
        y_valid: Validation labels.
        num_classes: Number of classes in the target variable (default is 2 for binary classification).
        n_calls: Number of iterations for hyperparameter tuning (default is 30).
        random_state: Random seed for reproducibility (default is 42).
    Returns:
        best_model: The best XGBoost model found during tuning.
        best_params: The best hyperparameters found during tuning.
        best_score: The log loss of the best model on the validation set.
    """
    param_space = {
        "learning_rate": Real(0.01, 0.5, prior='log-uniform'),
        "max_depth": Integer(3, 12),
        "min_child_weight": Real(1, 10),
        "gamma": Real(0, 5),
        "reg_alpha": Real(1e-3, 100.0, prior='log-uniform'),
        "reg_lambda": Real(1e-3, 100.0, prior='log-uniform'),
    }

    dimensions = list(param_space.values())
    param_names = list(param_space.keys())

    if num_classes > 2:
        eval_metric = "mlogloss"
        xgb_objective = "multi:softprob"
    else:
        eval_metric = "logloss"
        xgb_objective = "binary:logistic"

    def objective(params_list):
        params = dict(zip(param_names, params_list))

        model = XGBClassifier(
            objective=xgb_objective,
            eval_metric=eval_metric,
            num_class=num_classes,
            n_estimators=500,
            early_stopping_rounds=10,
            random_state=random_state,
            verbosity=0,
            **params
        )
        model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_valid, y_valid)], 
            verbose=False
            )
        y_proba = model.predict_proba(X_valid)
        loss = log_loss(y_valid, y_proba)

        return loss

    def on_iteration(res):
        i = len(res.x_iters)  # current iteration number
        # print(f"Completed iteration {i}, best log loss so far: {res.fun:.5f}")

    res = gp_minimize(
        func=objective, 
        dimensions=dimensions, 
        n_calls=n_calls,
        random_state=random_state,
        callback=[on_iteration],
        )
    
    best_params = dict(zip(param_names, res.x))
    # print(f"Best parameters: {best_params}", flush=True)

    best_model = XGBClassifier(
        objective=xgb_objective,
        eval_metric=eval_metric,
        use_label_encoder=False,
        num_class=num_classes,
        n_estimators=500,
        early_stopping_rounds=10,
        random_state=random_state,
        verbosity=0,
        **best_params
    )
    best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    best_score = log_loss(y_valid, best_model.predict_proba(X_valid))

    return best_model, best_params, best_score
