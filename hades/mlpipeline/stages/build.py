def process(config):
    try:
        model_type = config["build:GET"]["data"]["summary"]["modelType"]

        if model_type == "regressor":
            model_type_name = "Regression"
        elif model_type == "classifier":
            model_type_name = "Binary Classifier"
        elif model_type == "multi_classifier":
            model_type_name = "Multi Classifier"
        else:
            model_type_name = "**Unknown**"

        cols = []

        for feature in config["transform:POST"]["data"]["features"]:
            filt = list(
                filter(
                    lambda col: col["id"] == feature["id"],
                    config["prepare:POST"]["data"]["cols"],
                )
            )
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
            # regressor, classifier, multi_classifier
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
