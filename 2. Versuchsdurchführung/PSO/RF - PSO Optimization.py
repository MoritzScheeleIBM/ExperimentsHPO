from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from hyperactive import Hyperactive, ParticleSwarmOptimizer

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)


def model(opt):
    knr = RandomForestRegressor(random_state = 42, n_estimators=opt["n_estimators"], max_depth=opt["max_depth"], max_features = opt["max_features"], bootstrap = opt["bootstrap"], min_samples_leaf = opt["min_samples_leaf"], min_samples_split = opt["min_samples_split"])
    scores = cross_val_score(knr, X_train, y_train, cv=5)
    score = scores.mean()

    return score


search_space = {
    "n_estimators": np.arange(200, 2000, 10),
    "max_depth": np.arange(10, 110, 11),
    "max_features" : ['auto', 'sqrt'],
    "bootstrap" : [True, False],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10]
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