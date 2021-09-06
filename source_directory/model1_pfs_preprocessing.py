    """Model for preprocessing data for time series Data.
    """


import pandas as pd
from sklearn.preprocessing import LabelEncoder


def item_cat_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """Editing item_category to merge with items later on.

    Args:
        df (pd.DataFrame): item_category.

    Returns:
        pd.DataFrame: item_category with addition of big_category feature translated into English.
    """

    df["big_category"] = [x.split("-")[0].strip() for x in df.item_category_name]
    rus_eng = {
    "Книги": "books",
    "Подарки": "present",
    "Игры": "games",
    "Игровые консоли": "game consoles",
    "Аксессуары": "accesories",
    "Программы": "programs",
    "Музыка": "music",
    "Кино": "cinema",
    "Карты оплаты": "gift_cards",
    "Игры PC": "pc_games",
    "Служебные": "services",
    "Доставка товара": "delivery",
    "Карты оплаты (Кино, Музыка, Игры)": "payment_cards",
    "Чистые носители (шпиль)": "cd",
    "Элементы питания": "battery",
    "Игры Android": "android_games",
    "Игры MAC": "mac_games",
    "Билеты (Цифра)": "tickets",
    "PC": "pc",
    "Чистые носители (штучные)": "dvd",
    }
    df['big_category'].replace(to_replace=rus_eng, inplace=True)
    return df


def create_full_items(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Merging item_cat and items to get a df with item id, item_category_id, and big_category.

    Args:
        df (pd.DataFrame): a = items and b = item_cat.

    Returns:
        pd.DataFrame: Returns full_items.
    """

    full_items = pd.merge(
    a.drop(columns="item_name"),
    b.drop(columns="item_category_name"),
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
    
    city = [x.split(" ")[0] for x in df['shop_name']]
    df["city"] = city
    df.loc[df["city"] == "!Якутск", "city"] = "Якутск"
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
    names = {
    "date_block_num": "date_block_num",
    "shop_id": "shop_id",
    "item_id": "item_id",
    "item_category_id": "item_category_id",
    "big_category": "big_category",
    "city": "city",
    ("item_cnt_month", 0): "item_cnt_month_0",
    ("item_cnt_month", 1): "item_cnt_month_1",
    ("item_cnt_month", 2): "item_cnt_month_2",
    ("item_cnt_month", 3): "item_cnt_month_3",
    ("item_cnt_month", 4): "item_cnt_month_4",
    ("item_cnt_month", 5): "item_cnt_month_5",
    ("item_cnt_month", 6): "item_cnt_month_6",
    ("item_cnt_month", 7): "item_cnt_month_7",
    ("item_cnt_month", 8): "item_cnt_month_8",
    ("item_cnt_month", 9): "item_cnt_month_9",
    ("item_cnt_month", 10): "item_cnt_month_10",
    ("item_cnt_month", 11): "item_cnt_month_11",
    ("item_cnt_month", 12): "item_cnt_month_12",
    ("item_cnt_month", 13): "item_cnt_month_13",
    ("item_cnt_month", 14): "item_cnt_month_14",
    ("item_cnt_month", 15): "item_cnt_month_15",
    ("item_cnt_month", 16): "item_cnt_month_16",
    ("item_cnt_month", 17): "item_cnt_month_17",
    ("item_cnt_month", 18): "item_cnt_month_18",
    ("item_cnt_month", 19): "item_cnt_month_19",
    ("item_cnt_month", 20): "item_cnt_month_20",
    ("item_cnt_month", 21): "item_cnt_month_21",
    ("item_cnt_month", 22): "item_cnt_month_22",
    ("item_cnt_month", 23): "item_cnt_month_23",
    ("item_cnt_month", 24): "item_cnt_month_24",
    ("item_cnt_month", 25): "item_cnt_month_25",
    ("item_cnt_month", 26): "item_cnt_month_26",
    ("item_cnt_month", 27): "item_cnt_month_27",
    ("item_cnt_month", 28): "item_cnt_month_28",
    ("item_cnt_month", 29): "item_cnt_month_29",
    ("item_cnt_month", 30): "item_cnt_month_30",
    ("item_cnt_month", 31): "item_cnt_month_31",
    ("item_cnt_month", 32): "item_cnt_month_32",
    ("item_cnt_month", 33): "item_cnt_month_33",
    }
    df.rename(columns=names,inplace=True)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the duplicates that have the same shop_id and item_id to decrease the amount of data.

    Args:
        df (pd.DataFrame): train and test.

    Returns:
        pd.DataFrame: train and test without duplicates.
    """

    df.drop_duplicates(subset=['shop_id','item_id'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df


def encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uses LabelEncoder from sklearn to encode the city and big_category values.

    Args:
        df (pd.DataFrame): train.

    Returns:
        pd.DataFrame: train with encoded features.
    """

    df["big_category"] = LabelEncoder().fit_transform(df, df['big_category'])
    df["city"] = LabelEncoder().fit_transform(df, df['city'])


def main():

    #Import csv files

    item_cat = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/item_categories.csv")
    items = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/items.csv")
    train = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/sales_train.csv")
    shops = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/shops.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/pfs/data/test.csv")

    #Preparing dfs to merge with train

    item_cat = item_cat_preparation(item_cat)
    full_items = create_full_items(items, item_cat)
    shops = shops_preparation(shops)
    train = train_preparation(train)

    #Merging

    full_train = pd.merge(train, full_items, on="item_id", how="left")
    full_train = pd.merge(full_train, shops.drop(columns="shop_name"), on="shop_id", how="left")

    #Creating X_train, y_train, and X_test

    X_train = full_train.iloc[:, :-1]
    y_train = full_train.iloc[:, -1]
    X_test = full_train.drop(columns=["item_cnt_month_0"])


    if __name__ == '__main__':
        main()