import tensorflow as tf

from art.attacks.evasion import ProjectedGradientDescent
import art.estimators.classification
import art
import sklearn
import sklearn.datasets
import sklearn.model_selection


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    x, y = sklearn.datasets.make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=2,
                                                n_repeated=0, n_classes=2)
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x,y, test_size=0.2)


    model = tf.keras.models.Sequential([ 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape = (20, )), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
                                    ])
    model.summary()

    model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

    classifier = art.estimators.classification.KerasClassifier(model=model)
    classifier.fit(train_x, train_y, nb_epochs=5)
    pred = classifier.predict(test_x)
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.5)
    x_test_adv = attack.generate(x=test_x)
    adv_predictions = classifier.predict(x_test_adv)
    print(pred)
    print(adv_predictions)
