import librosa
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .audioclip import Clip


def add_subplot_axes(ax: Axes, position):
    box = ax.get_position()

    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]

    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]], axisbg='w')


def plot_clip_overview(clip: Clip, ax: Axes):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    with clip.audio as audio:
        ax_waveform.plot(np.arange(0, len(audio.raw)) / float(Clip.RATE), audio.raw)
        ax_waveform.get_xaxis().set_visible(False)
        ax_waveform.get_yaxis().set_visible(False)
        ax_waveform.set_title('{0} \n {1}'.format(clip.category, clip.filename), {'fontsize': 8}, y=1.03)

        librosa.display.specshow(clip.logamplitude, sr=Clip.RATE, x_axis='time', y_axis='mel', cmap='RdBu_r')
        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)


def plot_single_clip(clip: Clip):
    col_names = list('MFCC_{}'.format(i) for i in range(np.shape(clip.mfcc)[1]))
    MFCC = pd.DataFrame(clip.mfcc[:, :], columns=col_names)

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.75])
    ax_mfcc.set_xlim(-400, 400)
    ax_zcr = add_subplot_axes(ax, [0.0, 0.85, 1.0, 0.05])
    ax_zcr.set_xlim(0.0, 1.0)

    plt.title('Feature distribution across frames of a single clip ({0} : {1})'.format(clip.category, clip.filename), y=1.5)
    sns.boxplot(MFCC, vert=False, order=list(reversed(MFCC.columns)), ax=ax_mfcc)
    sns.boxplot(pd.DataFrame(clip.zcr, columns=['ZCR']), vert=False, ax=ax_zcr)


def plot_feature_one_clip(feature: np.ndarray, title: str, ax: Axes) -> None:
    sns.despine()
    ax.set_title(title, y=1.03)
    sns.displot(feature, ax=ax, bins=20, kde=True, rug=False,
                hist_kws={'histtype': 'stepfilled', 'alpha': 0.5},
                kde_kws={'shade': False},
                color=sns.color_palette('muted', 4)[2])


def plot_feature_all_clips(feature: np.ndarray, title: str, ax: Axes) -> None:
    sns.despine()
    ax.set_title(title, y=1.03)
    sns.boxplot(feature, ax=ax, vert=False,
                order=list(reversed(feature.columns)))


def plot_feature_aggregate(feature: np.ndarray, title: str, ax: Axes) -> None:
    sns.despine()
    ax.set_title(title, y=1.03)
    sns.displot(feature, ax=ax, bins=20, kde=True, rug=False,
                hist_kws={'histtype': 'stepfilled', 'alpha': 0.5},
                kde_kws={'shade': False},
                color=sns.color_palette('muted', 4)[1])


def generate_feature_summary(clips, category, clip, coeff) -> None:
    title = f"{clips[category][clip].category} : {clips[category][clip].filename}"
    MFCC = pd.DataFrame()
    aggregate = []
    for i in range(0, len(clips[category])):
        MFCC[i] = clips[category][i].mfcc[:, coeff]
        aggregate = np.concatenate([aggregate, clips[category][i].mfcc[:, coeff]])

    f = plt.figure(figsize=(14, 12))
    f.subplots_adjust(hspace=0.6, wspace=0.3)

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)

    ax1.set_xlim(0.0, 0.5)
    ax2.set_xlim(-100, 250)
    ax4.set_xlim(-100, 250)

    plot_feature_one_clip(clips[category][clip].zcr, f'ZCR distribution across frames\n{title}', ax1)
    plot_feature_one_clip(clips[category][clip].mfcc[:, coeff], f'MFCC_{coeff} distribution across frames\n{title}', ax2)

    clipcat = clips[category][clip].category
    plot_feature_all_clips(MFCC, f'Differences in MFCC_{coeff} distribution\nbetween clips of {clipcat}', ax3)
    plot_feature_all_clips(aggregate, f'Aggregate MFCC_{coeff} distribution\n(bag-of-frames across all clips\nof {clipcat})', ax4)


def plot_all_features_aggregate(clips, ax):
    ax_mfcc = add_subplot_axes(ax, [0.0, 0.0, 0.85, 1.0])
    ax_zcr = add_subplot_axes(ax, [0.9, 0.0, 0.1, 1.0])

    sns.set_style('ticks')

    col_names = [f'MFCC_{i}' for i in range(np.shape(clips[0].mfcc)[1])]
    aggregated_mfcc = pd.DataFrame(clips[0].mfcc[:, :], columns=col_names)

    for clip in clips:
        aggregated_mfcc = aggregated_mfcc.append(pd.DataFrame(clip.mfcc[:, :], columns=col_names))

    aggregated_zcr = pd.DataFrame(clips[0].zcr, columns=['ZCR'])
    for clip in clips:
        aggregated_zcr = aggregated_zcr.append(pd.DataFrame(clip.zcr, columns=['ZCR']))

    sns.despine(ax=ax_mfcc)
    ax.set_title('Aggregate distribution: {0}'.format(clips[0].category), y=1.10, fontsize=10)
    sns.boxplot(aggregated_mfcc, vert=True, order=aggregated_mfcc.columns, ax=ax_mfcc)
    ax_mfcc.set_xticklabels(range(0, 13), rotation=90, fontsize=8)
    ax_mfcc.set_xlabel('MFCC', fontsize=8)
    ax_mfcc.set_ylim(-150, 200)
    ax_mfcc.set_yticks((-150, -100, -50, 0, 50, 100, 150, 200))
    ax_mfcc.set_yticklabels(('-150', '', '', '0', '', '', '', '200'))

    sns.despine(ax=ax_zcr, right=False, left=True)
    sns.boxplot(aggregated_zcr, vert=True, order=aggregated_zcr.columns, ax=ax_zcr)
    ax_zcr.set_ylim(0.0, 0.5)
    ax_zcr.set_yticks((0.0, 0.25, 0.5))
    ax_zcr.set_yticklabels(('0.0', '', '0.5'))


def plot_features_scatter(feature1, feature2, category, category_name, ax, legend='small',
                          pretty_labels=None, font=None, crop_right=None):
    if font is None:
        font = matplotlib.font_manager.FontProperties()

    sns.despine()
    category_count = len(clip_features['category'].unique())
    colors = sb.color_palette("Set3", 10)
    plots = []
    labels = []
    markers = [
        (3, 0, 0),     # Triangle up
        (4, 0, 0),     # Rotated square
        (3, 0, 180),   # Triangle down
        (3, 0, 270),   # Triangle right
        (6, 1, 0),     # Star (6)
        (3, 0, 90),    # Triangle left
        (5, 1, 0),     # Star (5)
        (4, 0, 45),    # Square
        (8, 1, 0),     # Star (8)
        (0, 3, 0)      # Circle
    ]

    for c in range(0, category_count):
        f1 = feature1[category == c]
        f2 = feature2[category == c]
        size = 50 if c != 9 else 35
        plots.append(ax.scatter(f1, f2, c=colors[c], s=size, marker=markers[c]))
        labels.append(category_name[category == c][0:1][0][6:])

    font.set_size(11)
    ax.set_xlabel(feature1.name, fontproperties=font)
    ax.set_ylabel(feature2.name, fontproperties=font)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.3)
    ax.set_xticklabels(ax.get_xticks(), fontproperties=font)
    ax.set_yticklabels(ax.get_yticks(), fontproperties=font)

    if crop_right is not None:
        ax.set_xlim(ax.get_xlim()[0], crop_right)

    if pretty_labels is not None:
        labels = pretty_labels

    if legend == 'small':
        ax.legend(plots, labels, ncol=2, loc='upper center', frameon=False, fancybox=False, borderpad=1.0, prop=font)
    elif legend == 'big':
        font.set_size(11)
        ax.legend(plots, labels, ncol=5, columnspacing=2, markerscale=1.5, loc='upper center', frameon=False, fancybox=False, borderpad=1.0, prop=font)


def plot_pca() -> None:
    pca = PCA(n_components=2)
    pca.fit(clip_features.loc[:, 'MFCC_1 mean':'ZCR std dev'])
    X = pca.transform(clip_features.loc[:, 'MFCC_1 mean':'ZCR std dev'])
    clip_features['First principal component'] = X[:, 0]
    clip_features['Second principal component'] = X[:, 1]

    f, axes = plt.subplots(1, 1, figsize=(14, 6))
    pretty_labels = ['Barking dog', 'Rain', 'Sea waves', 'Crying baby', 'Ticking clock', 'Sneeze', 'Helicopter', 'Chainsaw', 'Crowing rooster', 'Crackling fire']

    plot_features_scatter(clip_features['First principal component'], clip_features['Second principal component'],
                        clip_features['category'], clip_features['category_name'],
                        axes, 'big', pretty_labels, crop_right=150)
