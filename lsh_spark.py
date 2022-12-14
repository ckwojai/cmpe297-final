from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH
import pandas as pd

train_raw = pd.read_csv('train.dat', header=None,
            names=["uid", "iid", "rating", "ts"], sep='\t', engine='python')
max_iid = 1699 # result from train_raw
u2irmap = {}
i2umap = {}
for _, r in train_raw.iterrows():
    uid, iid = int(r.uid), int(r.iid)
    if uid not in u2irmap:
        u2irmap[uid] = {}
    u2irmap[uid][iid] = int(r.rating)
    if iid not in i2umap:
        i2umap[iid] = set()
    i2umap[iid].add(uid)


test_raw = pd.read_csv('test.dat', header=None,
            names=["uid", "iid"], sep='\t', engine='python')
test = test_raw.values.tolist()

# expect to have:
# maxiid
# u2irmap (user-to-item-ratings map), and
# i2umap (item-to-users map)
def predict_rating(itemid, userhistory, num_neighbors=10, num_hash_tables=20, max_iid=1699):
    if itemid not in i2umap or len(userhistory) == 0: # no rating has given for this item, can't predict
        return None
    userDs = [(int(uid), Vectors.sparse(max_iid+1, u2irmap[uid])) for uid in u2irmap if uid in i2umap[itemid]]
    dfUsers = spark.createDataFrame(userDs, ["uid", "features"])
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=num_hash_tables)
    model = mh.fit(dfUsers)
    key = Vectors.sparse(max_iid+1, userhistory) # item_history expect to be a dict {<item>: rating}
    rows = model.approxNearestNeighbors(dfUsers, key, num_neighbors)
    ratings = []
    for r in rows.collect():
        ratings.append(u2irmap[r["uid"]][itemid])
    return int(round(sum(ratings) / len(ratings))) if ratings else None

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

import time
start_time = time.time()
ratings = []
for index, (uid, iid) in enumerate(test):
    print(f"Predicting entry {index}/{len(test)}")
    ratings.append(predict_rating(iid, u2irmap[uid]))
print("--- %s seconds ---" % (time.time() - start_time))

with open("test_lsh.dat" ,'w') as f:
    for r in ratings:
        if r:
            f.write(f"{r}\n")
        else:
            f.write(f"{3}\n")
