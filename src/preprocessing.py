import pandas as pd


def preprocess(raw_df, age_medians_by_title, combined_ticket_counts, fare_medians_by_pclass):
    d = raw_df.copy()

    d["IsFemale"] = d["Sex"] == "female"
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1

    d["Title"] = d["Name"].str.extract(r",\s*([^.]+)\.")
    d.loc[~d["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), "Title"] = "Rare"

    d["Age"] = d["Age"].fillna(
        d.set_index(["Title", "Sex"]).index.map(age_medians_by_title_sex)
    )    
    d["Embarked"] = d["Embarked"].fillna("S")
    d["HasCabin"] = d["Cabin"].notna()
    d["TicketGroupSize"] = d["Ticket"].map(combined_ticket_counts)

    d = pd.get_dummies(d, columns=["Title"], prefix="Title")
    d = pd.get_dummies(d, columns=["Embarked"], prefix="Embarked")

    return d