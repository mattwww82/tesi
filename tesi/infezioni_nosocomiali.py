import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# Caricare il dataset
tabella = pd.read_csv("infezioni_nosocomiali.csv")
#print(data.head())

# Trasformare 'sesso' in numerico
le = LabelEncoder()
tabella['sesso'] = le.fit_transform(tabella['sesso'])  # M=1, F=0

# Separare input e target
X = tabella.drop("infezione", axis=1)
y = tabella["infezione"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Creazione del modello
model = DecisionTreeClassifier()

# Addestramento
model.fit(X_train, y_train)


# Convertire 'sesso' in numerico
nuovo_paziente = [[65, 1, 18, 1, 1, 1, 1, 0]]  # M=1

# Predizione
predizione = model.predict(nuovo_paziente)
print("Predizione infezione:")  # 1 = sì, 0 = no
if predizione[0] == 1: print("il paziente è ad alto rischio d'infezione")
else:
    print("il paziente non è ad alto rischio infezione")

