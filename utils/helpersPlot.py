import matplotlib.pyplot as plt


def exibe_correlacao(df_corr):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(df_corr, interpolation='nearest')

    plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=45)
    plt.yticks(range(len(df_corr.index)), df_corr.index)

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig


def exibe_ambiguitys(dic_ambiguitys):
    fig, ax = plt.subplots(figsize= (8, 6))
    ax.bar(dic_ambiguitys.keys(), dic_ambiguitys.values())
    ax.set_title("Ambiguidade por Modelo")
    ax.set_xlabel("Modelo", fontsize= 12)
    ax.set_ylabel("Ambiguidade", fontsize= 12)
    ax.grid(axis= 'y', linestyle= '--', alpha= 0.7)
    plt.xticks(rotation=45)

    return fig
