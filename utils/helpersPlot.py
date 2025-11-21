import matplotlib.pyplot as plt


def exibe_correlacao(df_corr):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(df_corr, interpolation='nearest')

    plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=45)
    plt.yticks(range(len(df_corr.index)), df_corr.index)

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig
