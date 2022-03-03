import dataset

if __name__ == "__main__":
    E13 = dataset.Dataset("datasets/human/E13")
    print(E13)

    TFP = dataset.Dataset("datasets/human/TFP")
    print(TFP)

    HUM = E13 + TFP
    print(HUM)