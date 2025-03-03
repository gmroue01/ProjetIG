from hyperimpute.plugins.imputers import Imputers
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd

# Charger les données
print("Chargement des données...")
df = pd.read_csv("Big_Data_Strokdem.csv", sep=";")

# Charger l'imputeur
print("Chargement de l'imputeur...")
imputer = Imputers().get(
    "hyperimpute",
    optimizer="hyperband",
    classifier_seed=["logistic_regression", "catboost", "xgboost", "random_forest"],
    regression_seed=[
        "linear_regression",
        "catboost_regressor",
        "xgboost_regressor",
        "random_forest_regressor",
    ],
    class_threshold=5,
    imputation_order=2,
    n_inner_iter=100,
    select_model_by_column=True,
    select_model_by_iteration=True,
    select_lazy=True,
    select_patience=5,
)

# Imputation des données
print("Imputation des données...")
x_imputed = imputer.fit_transform(df)

# Sauvegarde des données imputées
print("Sauvegarde des données...")
x_imputed.dataframe().to_csv("Big_Data_Strokdem_imputed.csv", index=False)

print("Imputation terminée !")
