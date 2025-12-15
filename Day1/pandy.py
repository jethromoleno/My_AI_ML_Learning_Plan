import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')
# pd.set_option('max_columns', 200)

df = pd.read_csv('AIML/Day1/datasets/titanic/train.csv')

# df['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']

df = df[['PassengerId', 'Survived', 'Pclass', 'Age', 'Parch','Fare', 'Embarked']].copy()

df= df.rename(columns={'PassengerId': 'ID'})

print(df['Fare'].value_counts()).head(5).plot()