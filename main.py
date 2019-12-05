from utilities import LoadImages


def main():


    path = "./camera_cal"
    loader = LoadImages(path)
    #loader.showAll()
    for (i,img) in enumerate(loader.dataIterable()):
        print(i)
    return



if __name__ == "__main__":
    main()
