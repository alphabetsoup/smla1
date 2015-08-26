
import pickle


def main():
    with open('data/train.pickle', 'rb') as sr:
        g = pickle.load(sr)

if __name__ == '__main__':
    main()
