"""
Methods for cleaning the data.
"""

def clean(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2
