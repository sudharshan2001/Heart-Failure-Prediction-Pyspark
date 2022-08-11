def split_data (df, split_prop=[0.8,0.2]):

    zeros = df.filter(df["label"]==0)
    ones = df.filter(df["label"]==1)

    train0, test0 = zeros.randomSplit(split_prop, seed=69)
    train1, test1 = ones.randomSplit(split_prop, seed=69)

    train_set = train0.union(train1)

    test_set = test0.union(test1)
    return train_set, test_set

