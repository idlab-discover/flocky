import pandas as pd

filename = "f25-e1metrics_averaging"
df = pd.read_csv(filename + ".csv")
df["index"] = df[["epoch", "iteration"]].max(axis=1)
df = df.query("epoch != -1")
#test = df[["epoch", "iteration"]].max(axis=1)
grouped  = df.groupby("index", as_index=False)

outlierstats1 = grouped.agg({"test_acc": "std"}).rename(columns={"test_acc": "test_acc_stdev"})
outlierstats2 = grouped.agg({"test_acc": "median"}).rename(columns={"test_acc": "test_acc_median"})
outlierstats = pd.merge(outlierstats1, outlierstats2, on="index")
#outlierstats.reset_index(name="epochiter")
outliersdf = pd.merge(df, outlierstats, on="index")
outliersdf.to_csv("outtest.csv")
cleaned = outliersdf.query("test_acc < test_acc_median + 2 * test_acc_stdev & test_acc > test_acc_median - 2 * test_acc_stdev")

grouped2 = cleaned.groupby("index")

filtered = grouped2.agg({ "test_acc": "min"}).rename(columns={"test_acc": "test_acc_min"})
filtered = filtered.join(grouped2.agg({"test_acc": "median"}).rename(columns={"test_acc": "test_acc_median"}))
filtered = filtered.join(grouped2.agg({"test_acc": "max"}).rename(columns={"test_acc": "test_acc_max"}))
filtered = filtered.join(grouped2.agg({"val_acc": "min"}).rename(columns={"val_acc": "val_acc_min"}))
filtered = filtered.join(grouped2.agg({"val_acc": "median"}).rename(columns={"val_acc": "val_acc_median"}))
filtered = filtered.join(grouped2.agg({"val_acc": "max"}).rename(columns={"val_acc": "val_acc_max"}))

filtered.to_csv(filename + "_proc.csv")