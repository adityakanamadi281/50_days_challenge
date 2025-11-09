import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import set_config
from sklearn.datasets import  make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline



steps=[("standard_scaler",StandardScaler()),
       ("classifier",LogisticRegression())]

pipe1=Pipeline(steps)

set_config(display="diagram")

x,y=make_classification(n_samples=1000)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pipe1.fit(x_train,y_train)

y_pred=pipe1.predict(x_test)


steps=[("scaling",StandardScaler()),
       ("PCA",PCA(n_components=3)),
       ("SVC", SVC())]


pipe2=Pipeline(steps)

pipe2.fit(x_train,y_train)

numeric_processor = Pipeline(
    steps=[("imputation_mean",SimpleImputer(missing_values=np.nan,strategy="mean")),
           ("scaler",StandardScaler())]
)


categorical_processor=Pipeline(
    steps=[("imputtion_constant",SimpleImputer(fill_value="missing",strategy="constant")),
           ("onehot",OneHotEncoder(handle_unknown="ignore"))]
)


preprocessor=ColumnTransformer(
    [("categorical",categorical_processor,["gender","city"]),
     ("numerical",numeric_processor,["age","height"])]
)


pipe=make_pipeline(preprocessor,LogisticRegression())

