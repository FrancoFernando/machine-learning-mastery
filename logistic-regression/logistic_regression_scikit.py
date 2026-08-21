# The same model with scikit-learn.
# LogisticRegression applies an L2 penalty by default, controlled by C.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(features, labels)
print("sklearn weights:", model.coef_)          # [[ 0.68 -0.68]]
print("sklearn bias:", model.intercept_)        # [0.]
new_review = [[3, 1]]  # "Taka bruk taka taka!"
print("Probability:", model.predict_proba(new_review))  # [[0.20 0.80]]
print("Prediction:", model.predict(new_review))         # [1]
