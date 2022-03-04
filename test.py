import dataset

if __name__ == "__main__":
    E13 = dataset.Dataset("datasets/human/E13")
    print(E13)

    TFP = dataset.Dataset("datasets/human/TFP")
    print(TFP)

    HUM = E13 + TFP
    HUM.name = "HUM"
    print(HUM)

    FSF = dataset.Dataset("datasets/fake/FSF")
    print(FSF)

    INT = dataset.Dataset("datasets/fake/INT")
    print(INT)
    
    TWT = dataset.Dataset("datasets/fake/TWT")
    print(TWT)

    FAK = FSF + INT + TWT
    FAK.name = "FAK"
    print(FAK)
    
    UFAK = FAK.undersample(HUM.size)
    UFAK.name = "UFAK"
    print(UFAK)

