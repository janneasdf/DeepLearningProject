"""
Vanilla CNN

"""
from keras.preprocessing.image import load_img

def main():
    """ Run the vanilla CNN on the images in data/ """
    from keras.models import Sequential
    from keras.layers import Dense, Activation

    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])

if __name__ == "__main__":
    main()
