from carbon.mlpipeline.utils import csvFileSelector, typeOfColumn1, uniqueColumnIdGenerator


def process(config):
    df = csvFileSelector(config)
    # df.head()

    # sample = df.sample(n=10)
    # print(sample.dtypes)

    uniqueColumnId = uniqueColumnIdGenerator(df)
    # print(uniqueColumnId)

    columnlist1 = []
    for _, keys in zip(range(df.shape[1]), df.columns):
        columnlist1.append(
            {
                "id": uniqueColumnId[keys],
                "name": keys,
                "type": typeOfColumn1(df[keys]),
                "sample": list(df[keys].sample(n=10).dropna()),
                "imputable": False,
            }
        )

    return {
        "stage": "prepare:GET",
        "data": {
            "rows": df.shape[0],
            "target": "null",
            "nextAvialableId": df.shape[1] + 1,
            "typeConversion": {
                ## from Number
                "Number": ["Category", "Number"],
                ## from Text
                "Text": ["Category", "Text"],
                ## from Timestamp
                "Timestamp": ["Category", "Timestamp"],
                ## from Category
                "Category": ["Text", "Category"],
            },
            "cols": columnlist1,
        },
    }


# pprint(consume_func(config))
