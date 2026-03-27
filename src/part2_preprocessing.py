import pandas as pd


def preprocess_data():
    pred_universe = pd.read_csv("src/data/pred_universe_raw.csv")
    arrest_events = pd.read_csv("src/data/arrest_events_raw.csv")

    pred_universe["arrest_date_univ"] = pd.to_datetime(pred_universe["arrest_date_univ"])
    arrest_events["arrest_date_event"] = pd.to_datetime(arrest_events["arrest_date_event"])

    df_arrests = pred_universe.merge(arrest_events, on="person_id", how="outer")

    y_list = []
    current_charge_felony_list = []
    num_fel_arrests_last_year_list = []

    for _, row in pred_universe.iterrows():
        person_id = row["person_id"]
        arrest_id = row["arrest_id"]
        current_date = row["arrest_date_univ"]

        person_events = arrest_events[arrest_events["person_id"] == person_id]

        current_event = arrest_events[arrest_events["arrest_id"] == arrest_id]
        current_charge_felony = 1 if (current_event["charge_degree"] == "felony").any() else 0
        current_charge_felony_list.append(current_charge_felony)

        future_felonies = person_events[
            (person_events["charge_degree"] == "felony") &
            (person_events["arrest_date_event"] >= current_date + pd.Timedelta(days=1)) &
            (person_events["arrest_date_event"] <= current_date + pd.Timedelta(days=365))
        ]
        y_list.append(1 if len(future_felonies) > 0 else 0)

        past_felonies = person_events[
            (person_events["charge_degree"] == "felony") &
            (person_events["arrest_date_event"] >= current_date - pd.Timedelta(days=365)) &
            (person_events["arrest_date_event"] <= current_date - pd.Timedelta(days=1))
        ]
        num_fel_arrests_last_year_list.append(len(past_felonies))

    pred_universe["y"] = y_list
    pred_universe["current_charge_felony"] = current_charge_felony_list
    pred_universe["num_fel_arrests_last_year"] = num_fel_arrests_last_year_list

    print("What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year?")
    print(pred_universe["y"].mean())

    print("What share of current charges are felonies?")
    print(pred_universe["current_charge_felony"].mean())

    print("What is the average number of felony arrests in the last year?")
    print(pred_universe["num_fel_arrests_last_year"].mean())

    print(pred_universe["num_fel_arrests_last_year"].mean())
    print(pred_universe.head())

    pred_universe.to_csv("src/data/df_arrests.csv", index=False)

    return pred_universe