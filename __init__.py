from inferer import *
import sys

if __name__ == "__main__":
    inferer = Classifier()
    for arg in sys.argv[1::]:
        result = inferer.infer(arg)
        reversed_result = {acc: plant for plant, acc in result.items()}
        max_acc = max(reversed_result.keys())
        # print(result)
        print(f"{reversed_result[max_acc]}: {max_acc}")
