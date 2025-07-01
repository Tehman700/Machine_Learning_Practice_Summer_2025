import pandas as pd
import time

# Load datasets
df_categories = pd.read_csv("D:/ML Dataset/category_tree.csv")
df_events = pd.read_csv("D:/ML Dataset/events.csv")
df_props1 = pd.read_csv("D:/ML Dataset/item_properties_part1.csv")
df_props2 = pd.read_csv("D:/ML Dataset/item_properties_part2.csv")



starting_from_here = time.time()


# We are going to combine these two datasets as they are same in nature just split due to large size

df_item_props = pd.concat([df_props1, df_props2])

# Below is the code where it drops those rows where rows doesn't correspond to categoryid
cleaned_first = df_item_props[df_item_props['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates()

# Below is the code for renaming the value to categoryid
cleaned_first.rename(columns={'value': 'categoryid'}, inplace=True)

# Merging this cleaned column to event on left that shows the full clean dataset
df_events = df_events.merge(cleaned_first, on='itemid', how='left')

# Just Printing to see the results
print(df_events.head(20))
print(df_events.shape)

df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')


# Now as we know that data we must perform a new operation on all users so that we can get how much views
# and how they are getting that views and on which item they spent most time and didn't but anything for example


user_stats = df_events.groupby('visitorid').agg(
    total_views=('event',lambda x: (x == 'view').sum()),
    total_adds=('event',lambda x: (x == 'addtocart').sum()),
    total_purchases=('event',lambda x: (x == 'transaction').sum()),
    unique_items=('itemid',pd.Series.nunique),
    unique_categories=('categoryid', pd.Series.nunique),
    first_time=('timestamp', 'min'),
    last_time=('timestamp', 'max'),
    total_events=('event', 'count')
).reset_index()

# Time-based features
user_stats['active_seconds'] = (user_stats['last_time'] - user_stats['first_time']).dt.total_seconds()
user_stats['avg_seconds_per_event'] = user_stats['active_seconds'] / user_stats['total_events'].replace(0, 1)

# Ratios
user_stats['view_to_cart'] = user_stats['total_views'] / user_stats['total_adds'].replace(0, 1)
user_stats['cart_to_purchase'] = user_stats['total_adds'] / user_stats['total_purchases'].replace(0, 1)


# # Rule-based abnormal behavior labeling mostly from chatgpt
user_stats['abnormal'] = 0
user_stats.loc[(user_stats['total_views'] > 50) & (user_stats['total_purchases'] == 0), 'abnormal'] = 1
user_stats.loc[(user_stats['avg_seconds_per_event'] < 2), 'abnormal'] = 1  # possible bot-like behavior
user_stats.loc[(user_stats['view_to_cart'] > 20), 'abnormal'] = 1  # many views, almost no interest
user_stats.loc[(user_stats['unique_categories'] > 50), 'abnormal'] = 1  # unusual browsing patterns

# Showing a final detailed sample of the features on which we will apply the model

print(user_stats[['visitorid', 'total_views', 'total_adds', 'total_purchases', 'unique_items', 'avg_seconds_per_event', 'abnormal']].head(10))


ending = time.time()

print(f"\n Time taken: {ending - starting_from_here:.2f} seconds")
