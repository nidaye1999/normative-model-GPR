import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


def plot(data, x_test, model, name, file):
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=[6.4, 7.2])

	y_hat, y_var = model.predict(x_test, return_cov=True)
	ax1.scatter(data["x"], data["y"], s=2)
	ax1.plot(x_test[:, 0], y_hat, c="r")
	ax2.plot(x_test[:, 0], np.diag(y_var))
	ax1.set_title("prediction", size=15)
	ax1.tick_params(labelsize=10)
	ax2.set_title("variance", size=15)
	ax2.tick_params(labelsize=10)

	y_hat, y_var = model.predict(data[["x"]], return_cov=True)
	ax3.scatter(data["x"], (data["y"] - y_hat) / np.sqrt(np.diag(y_var)), s=2)
	ax3.set_title("z-score", size=15)
	ax3.tick_params(labelsize=10)
	plt.suptitle(name, size=17)
	plt.subplots_adjust(top=0.9)
	plt.savefig(file + ".png", dpi=300)
	plt.close()


"""Dataset 1"""
x_test = np.expand_dims(np.linspace(-3, 3, 301), axis=1)
n = 1000
rho, sigma = 0.75, 0.05
np.random.seed(20)

cov = np.array([[1, rho], [rho, 1]])
cholesky = np.linalg.cholesky(cov)
data = pd.DataFrame(scale(cholesky @ np.random.normal(size=(2, n)), axis=1).T, columns=["x", "y"])
data["y"] += np.random.normal(scale=sigma, size=n)
data -= data.mean()

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 1 (Original)", size=17)
plt.tight_layout()
plt.savefig("set_1.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 1; Original)", "linear_1")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 1; Original)", "rbf_1")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 1 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("set_1_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 1; Undersampled)", "linear_1_un")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 1; Undersampled)", "rbf_1_un")

"""Dataset 2"""
x_test = np.expand_dims(np.linspace(-3, 3, 301), axis=1)
n = 1000
rho, sigma = 0.75, 0.05
np.random.seed(20)

cov = np.array([[1, rho], [rho, 1]])
cholesky = np.linalg.cholesky(cov)
data = pd.DataFrame(scale(cholesky @ np.random.normal(size=(2, n)), axis=1).T, columns=["x", "y"])
data["y"] += np.random.normal(scale=sigma, size=n)
data -= data.mean()

data = pd.concat([data[data["x"] < 0] - [data["x"].min(), data["x"].min()], data[data["x"] >= 0] - [data["x"].max(), data["x"].max()]], axis=0)
fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 2 (Original)", size=17)
plt.tight_layout()
plt.savefig("set_2.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 2; Original)", "linear_2")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 2; Original)", "rbf_2")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 2 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("set_2_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 2; Undersampled)", "linear_2_un")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 2; Undersampled)", "rbf_2_un")

"""Dataset 3"""
x_test = np.expand_dims(np.linspace(-np.pi, np.pi, 301), axis=1)
n = 1000
sigma = 0.05
np.random.seed(20)

data = pd.DataFrame(np.random.uniform(-np.pi, np.pi, (n, 2)), columns=["x", "y"])
data["y"] += np.random.normal(scale=sigma, size=n)
data -= data.mean()

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 3 (Original)", size=17)
plt.tight_layout()
plt.savefig("set_3.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 3; Original)", "linear_3")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 3; Original)", "rbf_3")

GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data[["x"]], data["y"])
print("Dataset 3; Original:", GPR_linear_rbf_white.kernel_)
print("Dataset 3; Original; y_res.var():", (data.y - GPR_linear_rbf_white.predict(data[["x"]])).var())
print("Dataset 3; Original; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data, x_test, GPR_linear_rbf_white, "Linear+RBF+White Kernel (Dataset 3; Original)", "linear_rbf_white_3")

data_appendix = data.copy()
data_appendix.y += data_appendix.x ** 2
GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data_appendix[["x"]], data_appendix["y"])
print("Dataset 3; Quadratic:", GPR_linear_rbf_white.kernel_)
print("Dataset 3; Quadratic; y_res.var():", (data_appendix.y - GPR_linear_rbf_white.predict(data_appendix[["x"]])).var())
print("Dataset 3; Quadratic; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data_appendix, x_test, GPR_linear_rbf_white, "Linear+RBF+White Kernel (Dataset 3; Quadratic)", "linear_rbf_white_3_quad")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 3 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("set_3_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 3; Undersampled)", "linear_3_un")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 3; Undersampled)", "rbf_3_un")

"""Dataset 4"""
x_test = np.expand_dims(np.linspace(-np.pi, np.pi, 301), axis=1)
n = 1000
sigma = 0.05
np.random.seed(20)

data = pd.DataFrame(np.random.uniform(-np.pi, np.pi, (n, 2)), columns=["x", "y"])
data["y"] *= (np.sin(data["x"]) / 2 + 1)
data["y"] += np.random.normal(scale=sigma, size=n)
data -= data.mean()

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 4 (Original)", size=17)
plt.tight_layout()
plt.savefig("set_4.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 4; Original)", "linear_4")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 4; Original)", "rbf_4")

GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data[["x"]], data["y"])
print("Dataset 4; Original:", GPR_linear_rbf_white.kernel_)
print("Dataset 4; Original; y_res.var():", (data.y - GPR_linear_rbf_white.predict(data[["x"]])).var())
print("Dataset 4; Original; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data, x_test, GPR_linear_rbf_white, "Linear+RBF+White Kernel (Dataset 4; Original)", "linear_rbf_white_4")

data_appendix = data.copy()
data_appendix.y += data_appendix.x ** 2
GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data_appendix[["x"]], data_appendix["y"])
print("Dataset 4; Quadratic:", GPR_linear_rbf_white.kernel_)
print("Dataset 4; Quadratic; y_res.var():", (data_appendix.y - GPR_linear_rbf_white.predict(data_appendix[["x"]])).var())
print("Dataset 4; Quadratic; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data_appendix, x_test, GPR_linear_rbf_white, "Linear+RBF+White Kernel (Dataset 4; Quadratic)", "linear_rbf_white_4_quad")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y")
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("Dataset 4 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("set_4_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "Linear Kernel (Dataset 4; Undersampled)", "linear_4_un")

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "RBF Kernel (Dataset 4; Undersampled)", "rbf_4_un")
