import pandas as pd
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

# Start H2O on your local machine
h2o.init()
ecg_data = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/anomaly/ecg_discord_test.csv")
print(ecg_data.shape)
print(ecg_data.types)
print(ecg_data.head())

train_ecg = ecg_data[:20:, :]
test_ecg = ecg_data[:23, :]


def plot_stacked_time_series(df, title):
    stacked = df.stack()
    stacked = stacked.reset_index()
    total = [data[0].values for name, data in stacked.groupby('level_0')]
    # pd.DataFrame({idx: pos for idx, pos in enumerate(total)}, index=stacked['level_1']).plot(title=title)
    pd.DataFrame({idx: pos for idx, pos in enumerate(total)}).plot(title=title)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()


plot_stacked_time_series(ecg_data.as_data_frame(), "ECG data set")


def plot_bidimensional(model, test, recon_error, layer, title):
    bidimensional_data = model.deepfeatures(test, layer).cbind(recon_error).as_data_frame()

    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()
    bidimensional_data.plot(kind='scatter',
                            x='DF.L{}.C1'.format(layer + 1),
                            y='DF.L{}.C2'.format(layer + 1),
                            s=500,
                            c='Reconstruction.MSE',
                            title=title,
                            ax=ax,
                            colormap=cmap)
    layer_column = 'DF.L{}.C'.format(layer + 1)
    columns = [layer_column + '1', layer_column + '2']
    for k, v in bidimensional_data[columns].iterrows():
        ax.annotate(k, v, size=20, verticalalignment='bottom', horizontalalignment='left')
    fig.canvas.draw()
    plt.show()


seed = 13
anomaly_model = H2OAutoEncoderEstimator(
    activation="Tanh",
    hidden=[50, 20, 2, 20, 50],
    epochs=100,
    # sparse=True,
    # l1=1e-5,
    seed=seed,
    reproducible=True)

anomaly_model.train(
    x=train_ecg.names,
    training_frame=train_ecg
)

recon_error = anomaly_model.anomaly(test_ecg)
plot_bidimensional(anomaly_model, test_ecg, recon_error, 2, "2D representation of data points seed {}".format(seed))

# plot_stacked_time_series(anomaly_model.predict(ecg_data).as_data_frame(), "Reconstructed test set")

print(anomaly_model)

plt.figure()
df = recon_error.as_data_frame(True)
df["sample_index"] = df.index
df.plot(kind="scatter", x="sample_index", y="Reconstruction.MSE",
        title="reconstruction error", s=500)

len(recon_error)

anomaly_model.deepfeatures(train_ecg, 1).as_data_frame()  # .plot(kind='scatter', x='DF.L2.C1', y='DF.L2.C2')

for seed in range(1, 6):
    model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=[50, 20, 2, 20, 50],
        epochs=100,
        # sparse=True,
        # l1=1e-5,
        seed=seed,
        reproducible=True)
    model.train(
        x=train_ecg.names,
        training_frame=train_ecg)

    recon_error = model.anomaly(test_ecg)
    plot_bidimensional(model, test_ecg, recon_error, 2, "2D representation of data points seed {}".format(seed))
    # compute average and variance of the 2 dimensions

model = H2OAutoEncoderEstimator(
    activation="Tanh",
    hidden=[50, 20, 2, 20, 50],
    epochs=100,
    # sparse=True,
    # l1=1e-5,
    seed=1,
    reproducible=True)
model.train(
    x=train_ecg.names,
    training_frame=train_ecg
)

recon_error = model.anomaly(test_ecg)
bidimensional_data = model.deepfeatures(test_ecg, 2).cbind(recon_error).as_data_frame()
print(bidimensional_data)
