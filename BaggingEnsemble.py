from pyexpat.errors import XML_ERROR_NOT_STANDALONE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split

thyroid_sick=fetch_datasets()["thyroid_sick"]
x=thyroid_sick.data
y=thyroid_sick.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)
smote=SMOTE()
x_train_resampled, y_train_resampled=smote.fit_resample(x_train,y_train)
clf=BaggingClassifier(base_estimator=LogisticRegression(),
                        n_estimators=10,
                        random_state=0)
model=clf.fit(x_train_resampled,y_train_resampled)

y_pred=clf.predict(x_test)
sns.set_context("pster")
disp=plot_confusion_matrix(clf,x_test,y_test,cmap='cividis',colorbar=False)