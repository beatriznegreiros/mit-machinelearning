import sys
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
sys.path.append("..")
from utils import *
from features import *
from svm import *

train_x, train_y, test_x, test_y = get_MNIST_data()

n_components = 10

train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)

train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

model = SVC(random_state=0, kernel='poly', degree=3)
model.fit(train_pca, train_y)
pred_test = model.predict(test_pca)
test_error = compute_test_error_svm(test_y, pred_test)
print(test_error)
