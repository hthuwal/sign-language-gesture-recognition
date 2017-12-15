import cv2
import os
from os.path import exists

hc = []


def convert(dataset):
    fcount = []
    count = 0
    rootPath = os.getcwd()
    majorData = os.path.join(os.getcwd(), "majorData")
    if (not exists(majorData)):
        os.makedirs(majorData)
    dataset = os.path.join(os.getcwd(), dataset)
    os.chdir(dataset)
    x = os.listdir(os.getcwd())

    for gesture in x:
        adhyan = gesture
        gesture = os.path.join(dataset, gesture)
        os.chdir(gesture)
        frames = os.path.join(majorData, adhyan)
        if(not os.path.exists(frames)):
            os.makedirs(frames)
        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in videos:
            count = count+1
            name = os.path.abspath(video)
            print name
            cap = cv2.VideoCapture(name)  # capturing input video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print frameCount
            fcount.append((frameCount, video))
            # fname.append(video)
            os.chdir(gesture)
            cap.release()
            cv2.destroyAllWindows()
    os.chdir(rootPath)
    fcount.sort(key=lambda x: x[0], reverse=True)
    f = open("out.txt", "wb")

    for (a, b) in fcount:
        x = str(a)+" : "+b

        print x
        f.write(x+"\n")
    f.close()
convert("test/")
