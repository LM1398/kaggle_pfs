"""Model for preprocessing data for time series Data.
"""

import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder


def item_cat_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """Editing item_category to merge with items later on.

    Args:
        df (pd.DataFrame): item_category.

    Returns:
        pd.DataFrame: item_category with addition of big_category feature translated into English.
    """
    with open(
        "/Users/leo/samurai/pfs/data/input/big_category_rus_eng.json"
    ) as json_file:
        trans = json.load(json_file)
    df["big_category"] = [x.split("-")[0].strip() for x in df.item_category_name]
    df["big_category"].replace(to_replace=trans, inplace=True)
    return df


def create_full_items(
    items: pd.DataFrame = items, item_cat: pd.DataFrame = item_cat
) -> pd.DataFrame:

    """Merging item_cat and items to get a df with item id, item_category_id, and big_category.

    Args:
        df (pd.DataFrame): items and item_cat.

    Returns:
        pd.DataFrame: Returns full_items.
    """

    full_items = pd.merge(
        items.drop(columns="item_name"),
        item_cat.drop(columns="item_category_name"),
        on="item_category_id",
    )
    return full_items


def shops_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """ Extracts first word from shop_name and adds it as city column in shops.
    Also fixes one of the typos in the shop names.

    Args:
        df (pd.DataFrame): shops

    Returns:
        pd.DataFrame: shops with city column
    """

    city = [x.split(" ")[0] for x in df["shop_name"]]
    df["city"] = city
    df.loc[df["city"] == "!Якутск", "city"] = "Якутск"
    df.drop(columns='shop_name',inplace=True)
    return df


def train_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """Uses pivot_table to add item_cnt per date_block_num as a feature (for all 33 weeks).
    Also fixes the names "('item_cnt_day'), x" because it return an error when using it for models.

    Args:
        df (pd.DataFrame): train.

    Returns:
        pd.DataFrame: full_train.
    """

    data = df.pivot_table(
        index=["shop_id", "item_id"],
        values=["item_cnt_day"],
        columns=["date_block_num"],
        aggfunc="sum",
    ).fillna(0)
    data.reset_index(inplace=True)
    data.rename(columns=({"item_cnt_day": "item_cnt_month"}), inplace=True)
    df = pd.merge(df, data, on=["shop_id", "item_id"], how="left")
    df.columns = (
        ["date_block_num", "shop_id", "item_id", "item_category_id", "big_category", "city"]
        + ["_".join(["item_cnt_month", str(block) for block in range(len(34))])])
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the duplicates that have the same shop_id and item_id to decrease the amount of data.

    Args:
        df (pd.DataFrame): train and test.

    Returns:
        pd.DataFrame: train and test without duplicates.
    """

    df.drop_duplicates(subset=["shop_id", "item_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uses LabelEncoder from sklearn to encode the city and big_category values.

    Args:
        df (pd.DataFrame): train.

    Returns:
        pd.DataFrame: train with encoded features.
    """

    df["big_category"] = LabelEncoder().fit_transform(df, df["big_category"])
    df["city"] = LabelEncoder().fit_transform(df, df["city"])


def main():

    # Import csv files

    item_cat = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/item_categories.csv")
    items = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/items.csv")
    train = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/sales_train.csv")
    shops = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/shops.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/test.csv")

    # Preparing dfs to merge with train

    item_cat = item_cat_preparation(item_cat)
    full_items = create_full_items(items, item_cat)
    shops = shops_preparation(shops)
    train = train_preparation(train)
    train = drop_duplicates(train)
    # Merging

    full_train = pd.merge(train, full_items, on="item_id", how="left")
    full_train = pd.merge(
        full_train, shops, on="shop_id", how="left"
    )

    # Creating X_train, y_train, and X_test

    X_train = full_train.iloc[:, :-1]
    y_train = full_train.iloc[:, -1]
    X_test = full_train.drop(columns=["item_cnt_month_0"])

    if __name__ == "__main__":
        main()
