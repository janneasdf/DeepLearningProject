"""
Vanilla CNN

"""


def main():
    """ """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "force_compile", help="force model recompilation", action="store_true")
    parser.add_argument(
        "input_data_dir", help="input data directory", nargs='?', default="data_mini")
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
    from keras import optimizers
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV

    print(".\n" * 5)
    print("args:", args)
    print(".\n" * 5, flush=True)

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
        # img = img.reshape((1,) + img.shape)
        return img

    print("--- Reading and preprocessing data ---", flush=True)
    X = [create_img_array(img_path) for img_path in os.listdir(image_dir)]
    X = np.array(X)

    # Read the labels and convert to one-hot
    print("--- Reading labels and converting to one-hot vectors ---", flush=True)
    labels_csv = np.genfromtxt(labels_path, dtype=None,
                               delimiter=',', names=True)
    labels = [l[1] for l in labels_csv]
    n_unique_classes = len(set(labels))
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    Y_onehot = np_utils.to_categorical(Y)

    # Split data into train and test data
    train_test_split_random_seed = 345
    X_validation, X_test, Y_validation, Y_test = train_test_split(
        X, Y_onehot, test_size=0.2, random_state=train_test_split_random_seed)

    print(".\n" * 5)
    print("Samples:", len(X))
    print("   Training:", len(X_validation))
    print("   Testing:", len(X_test))
    print("Classes:", n_unique_classes)
    print(".\n" * 5, flush=True)

    # Create the model
    def vanilla_model():
        model_filename = 'vanilla.h5'
        model_path = 'cache/'
        if not args.force_compile:
            saved_model_exists = model_filename in os.listdir(model_path)
            if saved_model_exists:
                print("Using precompiled model", model_filename, flush=True)
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

        print("Compiling vanilla model", flush=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        save_model(model, os.path.join(model_path, model_filename))
        return model

    def transfer_model():
        """ Load up a model trained on ImageNet data and add extra layers """
        base_model = inception_v3.InceptionV3(
            weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(n_unique_classes, activation='softmax')(x)
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
    model_to_use = vanilla_model if args.model == 'vanilla' else transfer_model

    def transfer_branch():
        def compute_bottleneck_values(X):
            """ Compute bottleneck values for the pretrained model on our data """
            model = inception_v3.InceptionV3(
                weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
            bottleneck_values = model.predict(X)
            return bottleneck_values

        # def train_top_model(bottleneck_values, train_labels):
        #     """ Train the added layers """
        #     model = create_top_model(bottleneck_values.shape[1:])
        #     model.fit(x=bottleneck_values, y=train_labels)
        #     # import code
        #     # code.interact(local=dict(globals(), **locals()))
        #     predictions = model.predict(bottleneck_values)

        validation_values_path = 'cache/bottleneck_values_validation_' + \
            args.input_data_dir + '.npy'
        test_values_path = 'cache/bottleneck_values_test_' + args.input_data_dir + '.npy'

        if os.path.exists(validation_values_path) and os.path.exists(test_values_path):
            print("--- Using cached bottleneck values ---")
        else:
            print("--- Computing bottleneck values ---")
            np.save(validation_values_path,
                    compute_bottleneck_values(X_validation))
            np.save(test_values_path, compute_bottleneck_values(X_test))

        # train_top_model(np.load(validation_values_path), Y_validation)

        def create_top_model(input_shape, learning_rate, momentum):
            model = Sequential()
            model.add(Flatten(input_shape=input_shape))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu')) # needed?
            model.add(Dense(n_unique_classes, activation='softmax'))
            model.compile(
                optimizer=optimizers.SGD(
                    lr=learning_rate, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])
            return model

        val_bottleneck_values = np.load(validation_values_path)
        test_bottleneck_values = np.load(test_values_path)
        bottleneck_values_shape = val_bottleneck_values.shape[1:]

        def model_fn(learning_rate, momentum): return create_top_model(
            bottleneck_values_shape, learning_rate, momentum)

        def model1(): return model_fn(0.01, 0.0)

        final_model = model1()
        evaluation = final_model.evaluate(test_bottleneck_values, Y_test)

        import code
        code.interact(local=dict(globals(), **locals()))

        # estimator = KerasClassifier(build_fn=model_fn)
        # grid_search = GridSearchCV(estimator=estimator, param_grid={
        #     "learning_rate": [0.001, 0.01, 0.1, 0.3],
        #     "momentum": [0.0, 0.2, 0.4, 0.6, 0.8],
        # })
        # grid_results = grid_search.fit(val_bottleneck_values, Y_validation)
        # print("Best score:", grid_results.best_score_)
        # print("Best params:", grid_results.best_params_)

    transfer_branch()

    # estimator = KerasClassifier(build_fn=model_to_use)
    # # kfold = KFold(n_splits=10)
    # # score = cross_val_score(classifier, X, Y_onehot, cv=kfold)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2)
    # estimator.fit(X_train, Y_train)
    # test_predictions = estimator.predict(X_test)
    # import code
    # code.interact(local=dict(globals(), **locals()))

    # # Train the model
    # print("--- Training model ---")
    # model = vanilla_model()
    # model.fit(X, Y_onehot)

    # # Test the model
    # print("--- Testing vanilla model ---")
    # code.interact(local=dict(globals(), **locals()))


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except:
    #     import sys
    #     print("Unexpected error: ", sys.exc_info()[0])
    #     import code
    #     code.interact(local=locals())
