from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from hyperactive import Hyperactive, ParticleSwarmOptimizer

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)


def model(opt):
    mlp = MLPClassifier(hidden_layer_sizes =opt["hidden_layer_sizes"], activation=opt["activation"], solver=opt["solver"], alpha=opt["alpha"], learning_rate=opt["learning_rate"])
    scores = cross_val_score(mlp, X_train, y_train, cv=5)
    score = scores.mean()

    return score


search_space = {
    "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
    "activation" : ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': np.arange(0, 2, 0.001),
    'learning_rate': ['constant','adaptive']

}


optimizer = ParticleSwarmOptimizer(
    inertia=0.4,
    cognitive_weight=0.7,
    social_weight=0.7,
    temp_weight=0.3,
    rand_rest_p=0.05,
)

hyper = Hyperactive()
hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
hyper.run()