import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import scale
from PIL import Image
import os

plt.rcParams["font.family"] = "Arial"
os.makedirs("plots", exist_ok=True)

def plot(data, x_test, model, name, file, y_ax1=None, y_ax2=None, y_ax3=None):
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=[6.4, 7.2])

	y_hat, y_var = model.predict(x_test, return_cov=True)
	ax1.scatter(data["x"], data["y"], s=2)
	ax1.plot(x_test[:, 0], y_hat, c="r")
	ax2.plot(x_test[:, 0], np.diag(y_var))
	ax1.set_title("prediction", size=15)
	ax1.tick_params(labelsize=10)
	ax1.set_xlim(-3.5, 3.5)
	if y_ax1:
		ax1.set_ylim(y_ax1[0], y_ax1[1])
	ax2.set_title("variance", size=15)
	ax2.tick_params(labelsize=10)
	if y_ax2:
		ax2.set_ylim(y_ax2[0], y_ax2[1])

	y_hat, y_var = model.predict(data[["x"]], return_cov=True)
	ax3.scatter(data["x"], (data["y"] - y_hat) / np.sqrt(np.diag(y_var)), s=2)
	ax3.set_title("z-score", size=15)
	ax3.tick_params(labelsize=10)
	if y_ax3:
		ax3.set_ylim(y_ax3[0], y_ax3[1])
	plt.suptitle(name, size=17)
	plt.subplots_adjust(top=0.9)
	plt.savefig(file + ".png", dpi=300)
	plt.close()


def plot_break_axis(data, x_test, model, name, file, y_ax1=None, y_ax2=None, y_ax3=None, y_ax4=None):
	fig = plt.figure(figsize=[6.4, 7.2])
	outer = gridspec.GridSpec(3, 1)

	y_hat, y_var = model.predict(x_test, return_cov=True)
	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
	ax1 = plt.Subplot(fig, inner[0])
	ax1.scatter(data["x"], data["y"], s=2)
	ax1.plot(x_test[:, 0], y_hat, c="r")
	ax1.set_title("prediction", size=15)
	ax1.tick_params(labelsize=10)
	ax1.set_xlim(-3.5, 3.5)
	ax1.set_ylim(-5, 15)
	fig.add_subplot(ax1)

	inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.05)
	ax2 = plt.Subplot(fig, inner[0])
	ax3 = plt.Subplot(fig, inner[1])
	ax2.plot(x_test[:, 0], np.diag(y_var))
	ax3.plot(x_test[:, 0], np.diag(y_var))
	ax2.set_xlim(-3.5, 3.5)
	ax3.set_xlim(-3.5, 3.5)
	ax2.set_ylim(3.657, 3.721)
	ax3.set_ylim(3.298, 3.354)
	ax2.spines.bottom.set_visible(False)
	ax3.spines.top.set_visible(False)
	# ax2.xaxis.tick_top()
	# ax2.tick_params(labeltop=False)  # don't put tick labels at the top
	ax2.set_xticks([])
	ax3.xaxis.tick_bottom()
	ax2.set_title("variance", size=15)

	d = .5  # proportion of vertical to horizontal extent of the slanted line
	kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
	              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
	ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
	ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)

	fig.add_subplot(ax2)
	fig.add_subplot(ax3)

	inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])
	ax4 = plt.Subplot(fig, inner[0])
	y_hat, y_var = model.predict(data[["x"]], return_cov=True)
	ax4.scatter(data["x"], (data["y"] - y_hat) / np.sqrt(np.diag(y_var)), s=2)
	ax4.set_title("z-score", size=15)
	ax4.tick_params(labelsize=10)
	ax4.set_xlim(-3.5, 3.5)
	ax4.set_ylim(-3, 3)
	fig.add_subplot(ax4)

	plt.tight_layout()
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

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("A: Dataset 1 (Original)", size=17)
plt.tight_layout()
plt.savefig("plots/set_1.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "A: Linear Kernel (Dataset 1; Original)", "plots/linear_1", (-5.5, 5.5), (0.002499, 0.002524), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "A: RBF Kernel (Dataset 1; Original)", "plots/rbf_1", (-5.5, 5.5), (0.00245, 0.0053), (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "A: Matern Kernel (Dataset 1; Original)", "plots/Matern_1", (-5.5, 5.5), (0.002, 0.022), (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "A: Rational-quadratic Kernel (Dataset 1; Original)", "plots/Rational_quadratic_1", (-5.5, 5.5), (0.0023, 0.0070), (-105, 105))

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("A: Dataset 1 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("plots/set_1_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "A: Linear Kernel (Dataset 1; Undersampled)", "plots/linear_1_un", (-5.5, 5.5), (0.00248, 0.00302), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "A: RBF Kernel (Dataset 1; Undersampled)", "plots/rbf_1_un", (-5.5, 5.5), None, (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "A: Matern Kernel (Dataset 1; Undersampled)", "plots/Matern_1_un", (-5.5, 5.5), None, (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "A: Rational-quadratic Kernel (Dataset 1; Undersampled)", "plots/Rational_quadratic_1_un", (-5.5, 5.5), None, (-105, 105))

"""Dataset 2"""
x_test = np.expand_dims(np.linspace(-3, 3, 301), axis=1)
n = 1000
rho, sigma = 0.75, 0.05
np.random.seed(20)

cov = np.array([[1, rho], [rho, 1]])
cholesky = np.linalg.cholesky(cov)
data = pd.DataFrame(scale(cholesky @ np.random.normal(size=(2, n)), axis=1).T, columns=["x", "y"])
data["y"] += np.random.normal(scale=sigma, size=n)

data = pd.concat([data[data["x"] < 0] - [data["x"].min(), data["x"].min()], data[data["x"] >= 0] - [data["x"].max(), data["x"].max()]], axis=0)
fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("B: Dataset 2 (Original)", size=17)
plt.tight_layout()
plt.savefig("plots/set_2.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "B: Linear Kernel (Dataset 2; Original)", "plots/linear_2", (-5.5, 5.5), (0.002499, 0.002524), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "B: RBF Kernel (Dataset 2; Original)", "plots/rbf_2", (-5.5, 5.5), (0.00245, 0.0053), (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "B: Matern Kernel (Dataset 2; Original)", "plots/Matern_2", (-5.5, 5.5), (0.002, 0.022), (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "B: Rational-quadratic Kernel (Dataset 2; Original)", "plots/Rational_quadratic_2", (-5.5, 5.5), (0.0023, 0.0070), (-105, 105))

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("B: Dataset 2 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("plots/set_2_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "B: Linear Kernel (Dataset 2; Undersampled)", "plots/linear_2_un", (-5.5, 5.5), (0.00248, 0.00302), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "B: RBF Kernel (Dataset 2; Undersampled)", "plots/rbf_2_un", (-5.5, 5.5), None, (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "B: Matern Kernel (Dataset 2; Undersampled)", "plots/Matern_2_un", (-5.5, 5.5), None, (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "B: Rational-quadratic Kernel (Dataset 2; Undersampled)", "plots/Rational_quadratic_2_un", (-5.5, 5.5), None, (-105, 105))

"""Dataset 3"""
x_test = np.expand_dims(np.linspace(-np.pi, np.pi, 301), axis=1)
n = 1000
sigma = 0.05
np.random.seed(20)

data = pd.DataFrame(np.random.uniform(-np.pi, np.pi, (n, 2)), columns=["x", "y"])
data["y"] += np.random.normal(scale=sigma, size=n)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("C: Dataset 3 (Original)", size=17)
plt.tight_layout()
plt.savefig("plots/set_3.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "C: Linear Kernel (Dataset 3; Original)", "plots/linear_3", (-5.5, 5.5), (0.002499, 0.002524), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "C: RBF Kernel (Dataset 3; Original)", "plots/rbf_3", (-5.5, 5.5), (0.00245, 0.0053), (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "C: Matern Kernel (Dataset 3; Original)", "plots/Matern_3", (-5.5, 5.5), (0.0025, 0.0036), (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "C: Rational-quadratic Kernel (Dataset 3; Original)", "plots/Rational_quadratic_3", (-5.5, 5.5), (0.0025, 0.0029), (-105, 105))

GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data[["x"]], data["y"])
print("Dataset 3; Original:", GPR_linear_rbf_white.kernel_)
print("Dataset 3; Original; y_res.var():", (data.y - GPR_linear_rbf_white.predict(data[["x"]])).var())
print("Dataset 3; Original; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data, x_test, GPR_linear_rbf_white, "A: Linear+RBF+White Kernel (Dataset 3; Original)", "plots/linear_rbf_white_3", (-5.5, 5.5), None, (-3, 3))

data_appendix = data.copy()
data_appendix.y += data_appendix.x ** 2
GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data_appendix[["x"]], data_appendix["y"])
print("Dataset 3; Quadratic:", GPR_linear_rbf_white.kernel_)
print("Dataset 3; Quadratic; y_res.var():", (data_appendix.y - GPR_linear_rbf_white.predict(data_appendix[["x"]])).var())
print("Dataset 3; Quadratic; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot_break_axis(data_appendix, x_test, GPR_linear_rbf_white, "A: Linear+RBF+White Kernel (Dataset 3; Quadratic)", "plots/linear_rbf_white_3_quad")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("C: Dataset 3 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("plots/set_3_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "C: Linear Kernel (Dataset 3; Undersampled)", "plots/linear_3_un", (-5.5, 5.5), (0.00248, 0.00302), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "C: RBF Kernel (Dataset 3; Undersampled)", "plots/rbf_3_un", (-5.5, 5.5), None, (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "C: Matern Kernel (Dataset 3; Undersampled)", "plots/Matern_3_un", (-5.5, 5.5), None, (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "C: Rational-quadratic Kernel (Dataset 3; Undersampled)", "plots/Rational_quadratic_3_un", (-5.5, 5.5), None, (-105, 105))

"""Dataset 4"""
x_test = np.expand_dims(np.linspace(-np.pi, np.pi, 301), axis=1)
n = 1000
sigma = 0.05
np.random.seed(20)

data = pd.DataFrame(np.random.uniform(-np.pi, np.pi, (n, 2)), columns=["x", "y"])
data["y"] *= (np.sin(data["x"]) / 2 + 1)
data["y"] += np.random.normal(scale=sigma, size=n)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("D: Dataset 4 (Original)", size=17)
plt.tight_layout()
plt.savefig("plots/set_4.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "D: Linear Kernel (Dataset 4; Original)", "plots/linear_4", (-5.5, 5.5), (0.002499, 0.002524), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "D: RBF Kernel (Dataset 4; Original)", "plots/rbf_4", (-5.5, 5.5), (0.00245, 0.0053), (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "D: Matern Kernel (Dataset 4; Original)", "plots/Matern_4", (-5.5, 5.5), (0.0025, 0.0036), (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "D: Rational-quadratic Kernel (Dataset 4; Original)", "plots/Rational_quadratic_4", (-5.5, 5.5), (0.0025, 0.0029), (-105, 105))

GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data[["x"]], data["y"])
print("Dataset 4; Original:", GPR_linear_rbf_white.kernel_)
print("Dataset 4; Original; y_res.var():", (data.y - GPR_linear_rbf_white.predict(data[["x"]])).var())
print("Dataset 4; Original; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot(data, x_test, GPR_linear_rbf_white, "B: Linear+RBF+White Kernel (Dataset 4; Original)", "plots/linear_rbf_white_4", (-5.5, 5.5), None, (-3, 3))

data_appendix = data.copy()
data_appendix.y += data_appendix.x ** 2
GPR_linear_rbf_white = GaussianProcessRegressor(kernel=ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * DotProduct(sigma_0=0, sigma_0_bounds="fixed") + ConstantKernel(constant_value_bounds=(1e-30, 1e5)) * RBF() + WhiteKernel(), alpha=0)
GPR_linear_rbf_white.fit(data_appendix[["x"]], data_appendix["y"])
print("Dataset 4; Quadratic:", GPR_linear_rbf_white.kernel_)
print("Dataset 4; Quadratic; y_res.var():", (data_appendix.y - GPR_linear_rbf_white.predict(data_appendix[["x"]])).var())
print("Dataset 4; Quadratic; likelihood", GPR_linear_rbf_white.log_marginal_likelihood_value_)
plot_break_axis(data_appendix, x_test, GPR_linear_rbf_white, "B: Linear+RBF+White Kernel (Dataset 4; Quadratic)", "plots/linear_rbf_white_4_quad")

data = data.sample(frac=0.05)

fig = sns.jointplot(data=data, x="x", y="y", ylim=[-5.5, 5.5], xlim=[-3.5, 3.5])
fig.set_axis_labels("x", "y", size=15)
fig.ax_joint.tick_params(labelsize=13)
plt.suptitle("D: Dataset 4 (Undersampled)", size=17)
plt.tight_layout()
plt.savefig("plots/set_4_un.png", dpi=300)
plt.close()

GPR_linear = GaussianProcessRegressor(kernel=DotProduct(sigma_0=0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_linear.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_linear, "D: Linear Kernel (Dataset 4; Undersampled)", "plots/linear_4_un", (-5.5, 5.5), (0.00248, 0.00302), (-105, 105))

GPR_rbf_fixed = GaussianProcessRegressor(kernel=RBF(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_rbf_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_rbf_fixed, "D: RBF Kernel (Dataset 4; Undersampled)", "plots/rbf_4_un", (-5.5, 5.5), None, (-105, 105))

GPR_Matern_fixed = GaussianProcessRegressor(kernel=Matern(length_scale_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_Matern_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_Matern_fixed, "D: Matern Kernel (Dataset 4; Undersampled)", "plots/Matern_4_un", (-5.5, 5.5), None, (-105, 105))

GPR_RationalQuadratic_fixed = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed") + WhiteKernel(noise_level=sigma ** 2, noise_level_bounds="fixed"), alpha=0)
GPR_RationalQuadratic_fixed.fit(data[["x"]], data["y"])
plot(data, x_test, GPR_RationalQuadratic_fixed, "D: Rational-quadratic Kernel (Dataset 4; Undersampled)", "plots/Rational_quadratic_4_un", (-5.5, 5.5), None, (-105, 105))

"""Combine figure and save as TIFF"""
images = [Image.open("plots/set_" + str(x) + ".png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig2.tif', dpi=(300, 300))

images = [Image.open("plots/set_" + str(x) + "_un.png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig3.tif', dpi=(300, 300))

images = [Image.open("plots/linear_" + str(x) + ".png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig4.tif', dpi=(300, 300))

images = [Image.open("plots/linear_" + str(x) + "_un.png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig5.tif', dpi=(300, 300))

images = [Image.open("plots/rbf_" + str(x) + ".png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig6.tif', dpi=(300, 300))

images = [Image.open("plots/rbf_" + str(x) + "_un.png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig7.tif', dpi=(300, 300))

images = [Image.open("plots/Matern_" + str(x) + ".png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig8.tif', dpi=(300, 300))

images = [Image.open("plots/Matern_" + str(x) + "_un.png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig9.tif', dpi=(300, 300))

images = [Image.open("plots/Rational_quadratic_" + str(x) + ".png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig10.tif', dpi=(300, 300))

images = [Image.open("plots/Rational_quadratic_" + str(x) + "_un.png") for x in range(1, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((2250, int(max_height / total_width * 2250)), Image.LANCZOS)
new_im.save('Fig11.tif', dpi=(300, 300))

images = [Image.open("plots/linear_rbf_white_" + str(x) + ".png") for x in range(3, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((1125, int(max_height / total_width * 1125)), Image.LANCZOS)
new_im.save('Fig12.tif', dpi=(300, 300))

images = [Image.open("plots/linear_rbf_white_" + str(x) + "_quad.png") for x in range(3, 5)]
widths, heights = zip(*(i.size for i in images))
total_width = sum(widths)
max_height = max(heights)
new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
	new_im.paste(im, (x_offset, 0))
	x_offset += im.size[0]
new_im = new_im.resize((1125, int(max_height / total_width * 1125)), Image.LANCZOS)
new_im.save('FigS1.tif', dpi=(300, 300))
