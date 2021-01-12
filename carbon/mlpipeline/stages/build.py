import re


def process(config):
    try:
        model_type_name = config["build:GET"]["data"]["summary"]["modelType"]
        regression_1 = re.compile("regression")
        regression_2 = re.compile("Regression")
        regression_3 = re.compile("regress")

        binary_1 = re.compile("Binary Calssification")
        binary_2 = re.compile("Binary Calssifier")
        binary_3 = re.compile("2-calssifier")
        # binary_4 = re.compile("Binary")
        # binary_5 = re.compile("binary")

        if (regression_1.search(model_type_name) is not None or
                regression_2.search(model_type_name) is not None or
                regression_3.search(model_type_name) is not None):
            model_type = "regression"
        elif (binary_1.search(model_type_name) is not None or binary_2.search(model_type_name) is not None or
              binary_3.search(model_type_name) is not None or binary_3.search(model_type_name) is not None or
              binary_3.search(model_type_name) is not None):
            model_type = "2-classifier"
        else:
            model_type = "n-classifier"

        cols = []

        for feature in config["transform:POST"]["data"]["features"]:
            filt = list(
                filter(
                    lambda col: col["id"] == feature["id"],
                    config["prepare:POST"]["data"]["cols"],
                ))
            res = {
                "id": feature["id"],
                "include": feature["include"],
                "name": feature["name"],
                "origin": feature["origin"],
                "type": feature["type"],
                "weight": feature["weight"],
                "imputable": filt[0]["imputable"],
            }
            cols.append(res)

        data = {
            "filepath": config["consume:POST"]["data"]["filepath"],
            "filename": config["consume:POST"]["data"]["filename"],
            "rows": config["build:GET"]["data"]["summary"]["rows"],
            "features": config["build:GET"]["data"]["summary"]["features"],
            "includedFeatures": config["build:GET"]["data"]["summary"]["includedFeatures"],
            "target_id": config["prepare:POST"]["data"]["target"],
            "target_name": config["build:GET"]["data"]["summary"]["target"],
            "sampleUsing": config["build:POST"]["data"]["sampleUsing"],
            "splitUsing": config["build:POST"]["data"]["splitUsing"],
            "optimizeUsing": config["build:POST"]["data"]["optimizeUsing"],
            "downsampling": config["build:POST"]["data"]["downsampling"],
            "cv": config["build:POST"]["data"]["cv"],
            # 2-classfier, n-classifier, regression
            "model_type": model_type,
            "model_type_name": model_type_name,
            "cols": cols,
        }
        final_config = {"stage": "finalconfig:GET", "data": data}
        # pprint(final_config)
        return final_config
    except Exception as ex:
        # import traceback

        # print(traceback.format_exc())
        print(ex)
        raise
