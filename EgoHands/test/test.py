import glob
import os

if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.rename("labels", "./temp")
        os.makedirs("labels")
    paths = glob.glob("./temp/*")

    for p in paths:
        with open(p) as f1:
            with open(os.path.join("labels", os.path.basename(p)), "w") as f2:
                while True:
                    l = f1.readline()
                    if not l: break
                    f2.write("0" + l[1:])

