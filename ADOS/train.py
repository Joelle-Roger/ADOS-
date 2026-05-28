import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# load dataset
df = pd.read_excel(r"D:\Uni Joelle\ADOS\Autism children data 200.xlsx")

print("Columns:",df.columns)

df = df.drop(columns=["First  Diagnosis"])

# ✔ encode Gender (Male/Female → 0/1)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

# ✔ encode target (Autism → 1, No Autism → 0)
df["Class"] = LabelEncoder().fit_transform(df["Class"])

X = df.drop("Class", axis=1)
y = df["Class"]
# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# test
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

counts = df["Class"].value_counts()

print("ASD count:", counts[1])
print("Non-ASD count:", counts[0])

# save model
joblib.dump(model, "model.pkl")