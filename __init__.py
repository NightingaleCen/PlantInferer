from inferer import *
import sys

if __name__ == "__main__":
    inferer = Classifier()
    for arg in sys.argv[1::]:
        print(inferer.infer(arg))
