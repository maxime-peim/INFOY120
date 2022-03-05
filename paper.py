import dataset

def build_paper_datasets():
    # building human dataset
    HUM = dataset.Dataset("datasets/human/E13",
                          "datasets/human/TFP")
    HUM.name = "HUM"

    # building fake dataset
    FAK = dataset.Dataset("datasets/fake/FSF",
                          "datasets/fake/INT",
                          "datasets/fake/TWT")
    FAK.name = "FAK"
    FAK.inplace_undersample(HUM.size)

    # building base dataset
    BAS = HUM + FAK
    BAS.name = "BAS"

    return HUM, FAK, BAS


if __name__ == "__main__":
    HUM, FAK, BAS = build_paper_datasets()

    print(HUM, FAK, BAS)