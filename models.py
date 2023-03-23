import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data.DataLoading import get_df









def main():
    df = get_df(filepath = "data\\seedling_labels_with_features.csv")
    expert_opinion_binary = np.array(df["expert_binary"]).reshape(-1,)
    features = df.columns[-6:]
    for feature in features:
        feature_vector = np.array(df[feature]).reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(feature_vector, expert_opinion_binary, test_size=0.20, random_state=0)
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        predictions_binary = logisticRegr.predict(np.array(df[feature]).reshape(-1, 1))
        predictions_prob = logisticRegr.predict_proba(np.array(df[feature]).reshape(-1, 1))[:,1]
        score = logisticRegr.score(x_test, y_test)
        print(f"The accuracy of the model {feature} {score}")
        model_name_prob = str(feature) + "_lr_probability"
        model_name_binary = str(feature) + "_lr_binary"
        df[model_name_prob] = predictions_prob
        df[model_name_binary] = predictions_binary

        df.to_csv("data\\seedling_labels_with_features_and_predictions.csv", index=False)
        df.to_excel ("data\\seedling_labels_with_features_and_predictions.xls", index=False)


if __name__ =="__main__":
    main()