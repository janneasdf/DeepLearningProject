"""
Vanilla CNN

"""

def main():
    """ """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("force_compile", help="force model recompilation", action="store_true")
    parser.add_argument("input_data_dir", help="input data directory", nargs='?', default="data_mini")
    parser.add_argument("model", help="model to use (vanilla, transfer)")
    args = parser.parse_args()

    # Imports
    import os
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from skimage import transform
    from keras.applications import inception_v3
    from keras.preprocessing.image import load_img, img_to_array
    from keras.models import Sequential, load_model, save_model, Model
    from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
    from keras.utils import np_utils
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score, KFold

    print(".\n" * 5)
    print("args:", args)
    print(".\n" * 5)

    # Set params
    data_dir = args.input_data_dir + "/"
    labels_path = data_dir + "labels.csv/labels.csv"
    image_dir = data_dir + "train/"
    image_size = (256, 256)

    # Read the training and test data
    def create_img_array(img_path):
        """ Read image from disk and preprocess it """
        img = img_to_array(load_img(os.path.join(image_dir, img_path)))
        img /= 255.0
        img = transform.resize(img, (image_size[0], image_size[1]), order=1)
        img = img_to_array(img)
        if args.model == 'transfer':
            img = inception_v3.preprocess_input(img)
        #img = img.reshape((1,) + img.shape)
        #img = transform.resize(img, (image_size[0], image_size[1], 3), order=1)
        return img

    print("--- Reading and preprocessing data ---")
    X = [create_img_array(img_path) for img_path in os.listdir(image_dir)]
    X = np.array(X)

    # Read the labels and convert to one-hot
    labels_csv = np.genfromtxt(labels_path, dtype=None,
                           delimiter=',', names=True)
    labels = [l[1] for l in labels_csv]
    n_unique_classes = len(set(labels))
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    Y_onehot = np_utils.to_categorical(Y)

    # Create the model
    def vanilla_model():
        model_filename = 'vanilla.h5'
        model_path = 'saved_models/'
        if not args.force_compile:
            saved_model_exists = model_filename in os.listdir(model_path)
            if saved_model_exists:
                print("Using precompiled model", model_filename)
                return load_model(os.path.join(model_path, model_filename))

        # TODO - customize model more
        # input_shape = (3, image_size[0], image_size[1])
        input_shape = (image_size[0], image_size[1], 3)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_unique_classes))
        model.add(Activation('sigmoid'))

        print("Compiling vanilla model")
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        save_model(model, os.path.join(model_path, model_filename))
        return model

    def transfer_model():
        """ Load up a model trained on ImageNet data and add extra layers """
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(n_unique_classes, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        # Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False
        print("Compiling transfer model")
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    print("--- Creating model ---")
    print("Model =", args.model)
    model_to_use = vanilla_model if args.model=='vanilla' else transfer_model
    estimator= KerasClassifier(build_fn=model_to_use)
    # kfold = KFold(n_splits=10)
    # score = cross_val_score(classifier, X, Y_onehot, cv=kfold)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2)
    estimator.fit(X_train, Y_train)
    test_predictions = estimator.predict(X_test)
    import code
    code.interact(local=dict(globals(), **locals()))

    # Train the model
    print("--- Training model ---")
    model = vanilla_model()
    model.fit(X, Y_onehot)

    # Test the model
    print("--- Testing vanilla model ---")
    code.interact(local=dict(globals(), **locals()))


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except:
    #     import sys
    #     print("Unexpected error: ", sys.exc_info()[0])
    #     import code
    #     code.interact(local=locals())
