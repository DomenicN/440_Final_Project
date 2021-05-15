"""
main.py - main file for 440 final project for RNA seq analysis
"""

# imports
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import matplotlib.pyplot as plt
import pickle
import mygene
from os import path
mg = mygene.MyGeneInfo()

# Global vars
# Initialize Babos dataset
babos_samples = {
    'GSM3964244_MEFs_': 'D0',
    'GSM3964245_6F_P4_': 'D4',
    'GSM3964246_DDRR_P4_': 'D4',
    'GSM3964247_6F_P8_': 'D8',
    'GSM3964248_DDRR_P8_': 'D8',
    'GSM3964249_6F_iMN1_': 'D14',
    'GSM3964250_6F_iMN2_': 'D14',
    'GSM3964251_DDRR_iMN1_': 'D14',
    'GSM3964252_DDRR_iMN2_': 'D14'
}

shie_samples = {
    'GSM2836267_D0.': 'D0',
    'GSM2836268_D2-1.': 'D2',
    'GSM2836269_D2-2.': 'D2',
    'GSM2836270_D4-1.': 'D4',
    'GSM2836271_D4-2.': 'D4',
    'GSM2836272_D6-1.': 'D6',
    'GSM2836273_D6-2.': 'D6',
    'GSM2836274_D8-1.': 'D8',
    'GSM2836275_D8-2.': 'D8',
    'GSM2836276_D9-1-2i.': 'D9',
    'GSM2836277_D9-2-2i.': 'D9',
    'GSM2836278_D10-1-2i.': 'D10',
    'GSM2836279_D10-2-2i.': 'D10',
    'GSM2836280_D11-2i.': 'D11',
    'GSM2836281_D12-1-2i.': 'D12',
    'GSM2836282_D12-2-2i.': 'D12',
    'GSM2836283_D16-2i.': 'D16',
    'GSM2836284_iPSCs-2i.': 'iPSCs',
}

fran_samples = ['GSE112004_counts.030U.tsv',
              'GSE112004_counts.031U.tsv',
              'GSE112004_counts.032U.tsv',
              'GSE112004_counts.033U.tsv',
              'GSE112004_counts.672T.tsv',
              'GSE112004_counts.673T.tsv',
              'GSE112004_counts.674T.tsv',
              'GSE112004_counts.675T.tsv',
              'GSE112004_counts.676T.tsv',
              'GSE112004_counts.677T.tsv',
              'GSE112004_counts.678T.tsv',
              'GSE112004_counts.679T.tsv',
              'GSE112004_counts.932S.tsv',
              'GSE112004_counts.933S.tsv',
              'GSE112004_counts.934S.tsv',
              'GSE112004_counts.935S.tsv',
              'GSE112004_counts.936S.tsv',
              'GSE112004_counts.937S.tsv',
              'GSE112004_counts.938S.tsv',
              'GSE112004_counts.939S.tsv']

SAMPLES = [babos_samples, shie_samples, fran_samples]
PATHS = ['Data/Babos/', 'Data/Shiebinger/GSE106340_RAW/', 'Data/Francesconi/']
NAMES = ['babos', 'shiebinger', 'francesconi']
META = [None, None, 'Data/Francesconi/GSE112004_series_matrix.txt']


def make_raw_dataset(samples, path, name):
    """
    Function to load, preprocess and concatenate a dataset from multiple RNAseq
     samples
    Inputs:
     samples, dictionary of sample file prefixes as keys and timepoint metadata
      as values
     path, path to directory containing sample files
     name, dataset name for labeling AnnData object metadata
    Output: AnnData object of concatenated samples, annotated with dataset,
     timepoint, and sample id labels
    """
    anndata_dict = {}

    for sm in samples.keys():
        print(sm)

        # read in data from GEO file
        data = sc.read_10x_mtx(path, prefix=sm, cache=True)

        # add metadata information
        data.obs['dataset'] = name
        data.obs['timepoint'] = samples[sm]

        # add to dict for concatenation
        anndata_dict[sm] = data

    # concatenate samples
    data_full = ad.concat(anndata_dict, join='outer', label='sample id', index_unique='_', fill_value=0.0)
    return data_full

def make_raw_dataset_tsv(samples, meta, path, name):
    """
    Gets anndata object when samples are in tsv format
    :param samples: list of sample prefixes
    :param meta: metadata file path
    :param path: path to data
    :param name: name of dataset
    :return: full anndata object
    """
    anndata_dict = {}
    metadata = get_francesconi_metadata(meta)
    for sm in samples:
        print(sm)
        full_path = path + sm

        # read data from geo file
        data = sc.read(full_path, cache=True)
        data = data.transpose()

        # add metadata info
        data.obs['dataset'] = name
        with open(full_path, 'r') as f:
            line = f.readline().split()
            # get first name
            n = line[0]
            time = metadata.loc[metadata['title'] == n, 'time'].to_string(index=False)
            treatment = metadata.loc[metadata['title'] == n, 'treatment'].to_string(index=False)
            time = time.replace('day ', 'D', 1)
        data.obs['timepoint'] = time
        if treatment == 'reprogramming' or time == '0h':
            anndata_dict[sm] = data

    # concatenate samples
    data_full = ad.concat(anndata_dict, join='outer', label='sample id', index_unique='_', fill_value=0.0)
    return rename_genes(data_full)

def run_normalization(dataset, n_top_genes=None, plot=False):
    """
    Runs normalization and filtering based on zheng17 paper
    :param dataset: AnnData dataset
    :return: filtered dataset
    """

    norm_data = sc.pp.recipe_zheng17(dataset,
                                     n_top_genes=n_top_genes,
                                     log=True,
                                     plot=plot)
    return norm_data

def run_harmony_integration(dataset, normalize=True):
    """
    Runs harmony integration and return dataframe

    :param dataset: anndata dataset to analyze
    :return: pandas dataframe with principal components and
    modified principal components
    """
    if normalize:
        run_normalization(dataset)
    sc.tl.pca(dataset)
    sce.pp.harmony_integrate(dataset, 'dataset')
    result_df = dataset.obsm.to_df()
    result_df['Sample ID'] = dataset.obs['sample id']
    ids = list(set(dataset.obs['sample id']))
    id_map = [(ids[i], i) for i in range(len(ids))]
    id_map = dict(id_map)
    result_df['color'] = [id_map[idx] for idx in dataset.obs['sample id']]
    return result_df

def states_across_time():
    """
    Makes tSNE plots across time for Babos and Shie
    :return:
    """
    # Analyze states across time
    states = [[({'GSM3964244_MEFs_': 'D0'}, 'Data/Babos/', 'babos'),
                      ({'GSM2836267_D0.': 'D0'}, 'Data/Shiebinger/GSE106340_RAW/', 'shiebinger')],
              [({'GSM3964245_6F_P4_': 'D4'}, 'Data/Babos/', 'babos'),
               ({'GSM2836270_D4-1.': 'D4'}, 'Data/Shiebinger/GSE106340_RAW/', 'shiebinger')],
              [({'GSM3964247_6F_P8_': 'D8'}, 'Data/Babos/', 'babos'),
               ({'GSM2836274_D8-1.': 'D8'}, 'Data/Shiebinger/GSE106340_RAW/', 'shiebinger')],
              [({'GSM3964249_6F_iMN1_': 'D14'}, 'Data/Babos/', 'babos'),
               ({'GSM2836288_iPSCs-serum.': 'iPSCs'}, 'Data/Shiebinger/GSE106340_RAW/', 'shiebinger')]]

    for state_data in states:
        raw_datasets = [make_raw_dataset(*sample) for sample in state_data]
        full_data = ad.concat(raw_datasets, join='outer', label='dataset')
        pca_df = run_harmony_integration(full_data)

        sc.tl.tsne(full_data, use_rep='X_pca_harmony')
        sc.pl.tsne(full_data, color='sample id')

def generate_dotplots(adatas, marker_dicts):
    """
    Generates dotplots for datasets
    :param adatas: iterable container of anndatasets
    :param marker_dicts: container of marker dictionaries
    :return:
    """
    n_samples = len(adatas)
    fig, ax = plt.subplots(1, n_samples)
    if n_samples > 1:
        for i in range(n_samples):
            sc.pl.dotplot(adatas[i], marker_dicts[i], groupby='timepoint',
                          ax=ax[i], show = False)
    else:
        sc.pl.dotplot(adatas[0], marker_dicts[0], groupby='timepoint',
                      ax=ax, show=False)

    plt.show()

def generate_neighbors_analysis(adata, filename):
    """
    Generates neighbors and pickles data
    :param adata: Anndata object of data
    :return:
    """
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20, use_rep='X_pca_harmony')
    with open(filename, 'wb') as f:
        pickle.dump(adata, f)

def get_francesconi_metadata(series_path):
    """
    Gets metadata for francesconi dataset
    :param series_path: path to series matrix for francesconi
    :return: metadata dataframe
    """
    with open(series_path, 'r') as f:
        for line in f.readlines():
            if len(line.split()) > 0:
                if line.split()[0] == '!Sample_title':
                    line = line.split()
                    fran_titles = line[1:]
                    fran_titles = [fran_title[1:-1] for fran_title in fran_titles]
                elif line.split()[0] == '!Sample_characteristics_ch1' and line.split()[1] == '"cell':
                    cell_types = line.split()
                    cell_types = [cell_types[i] for i in range(len(cell_types)) if i % 4 == 3]
                elif line.split()[0] == '!Sample_characteristics_ch1' and line.split()[1]== '"treatment:':
                    treatments = line.split()[1:]
                    treatments = treatments[1:len(treatments):2]
                    treatments = [treatment[:-1] for treatment in treatments]
                elif line.split()[0] == '!Sample_characteristics_ch1' and line.split()[1] == '"time':
                    times = line.split('\t"time point: ')[1:]
                    times = [time[:-1] for time in times]
                    times[-1] = times[-1][:-1]

        data = {'title':fran_titles, 'cell type':cell_types, 'treatment':treatments, 'time':times}
        return pd.DataFrame(data)

def rename_genes(data_ad):
    """
    Renames genes from ensembl id to gene name
    :param data_ad: anndata frame with data
    :return: odified anndata frame
    """
    # manually add in genes missing from database
    extras = pd.read_csv('./missing_gene_ids.csv', header=None)
    extras_d = dict(zip(extras.loc[:, 0], extras.loc[:, 1]))
    # create list of renamed genes
    gene_symbols = []
    to_remove = []
    added = []
    # remove version number from emsembl id
    gene_ensembl = [g[:g.find('.')] for g in data_ad.var_names[0:]]
    # convert ensembl id to gene symbol
    gene_info = mg.getgenes(gene_ensembl, species='mouse', fields='symbol')
    for g in gene_info:
        if not g['query'] in added:  # check for duplicates in queries
            # if not in mygene database, use gene symbol from extras file
            if 'symbol' not in g.keys():
                if g['query'] in extras_d.keys():
                    gene_symbols.append(extras_d[g['query']])
                    to_remove.append(extras_d[g['query']])
                # if not in extras file, leave name as is
                else:
                    gene_symbols.append(g['query'])
                    to_remove.append(g['query'])
            # else use gene symbol from mygene
            else:
                gene_symbols.append(g.get('symbol', g['query']))

            added.append(g['query'])  # check for duplicates in queries
    renamed_data_ad = data_ad
    renamed_data_ad.var_names = gene_symbols
    renamed_data_ad.var_names_make_unique()
    renamed_data_ad = renamed_data_ad[:, ~np.isin(renamed_data_ad.var.index.values, to_remove)]
    return renamed_data_ad

def make_all_raw_datasets(samples, paths, names, meta):
    """
    reads all datasets and performs integration
    :param samples: list of samples
    :param paths: list of paths
    :param names: list of names
    :param meta: list of metadata locations
    :return:
    """
    datasets = []
    for i in range(len(meta)):
        # make raw datsets using helper functions
        if meta[i] is None:
            dataset = make_raw_dataset(samples[i], paths[i], names[i])
            sc.pp.filter_genes(dataset, min_cells=10)
            run_normalization(dataset, n_top_genes=10000)
            datasets.append(dataset)
        else:
            dataset = make_raw_dataset_tsv(samples[i], meta[i], paths[i], names[i])
            sc.pp.filter_genes(dataset, min_cells=10)
            run_normalization(dataset, n_top_genes=10000)
            datasets.append(dataset)
    # concatenate data
    all_data = ad.concat(datasets, join='outer', label='sample id', index_unique='_', fill_value=0.0)

    # run harmony
    run_harmony_integration(all_data, normalize=False)

    # save data to reduce computation time
    with open('integrated/all_integrated', 'wb') as f:
        pickle.dump(all_data, f)

    datasets_integrated = []
    for name in names:
        dataset_int = all_data[np.equal(all_data.obs['dataset'], name), :]
        name_str = 'integrated/' + name + '_integrated'
        with open(name_str, 'wb') as f:
            pickle.dump(dataset_int, f)
        datasets_integrated.append(dataset_int)
    return all_data, datasets_integrated

def plot_genes_on_trajectories(datasets, hits):
    """
    Plots gene expression on trajectories
    :param datasets: list of anndata objects
    :param hits: list of lists of gene hits
    :return:
    """
    for dataset in datasets[:len(hits)]:
        sc.pl.paga(dataset, threshold=0.0, show=True)
    for i in range(len(hits)):
        for hit in hits[i]:
            sc.pl.draw_graph(datasets[i], color=hit, legend_loc='on data')

def make_tsne_plots(datasets):
    """
    Plots tsne for all datasets colored by timepoint
    :param datasets: List of anndata objects
    :return:
    """
    n_samples = len(datasets)
    fig, ax = plt.subplots(1, n_samples)

    for i in range(n_samples):
        dataset = datasets[i]
        sc.tl.tsne(dataset, use_rep='X_pca_harmony')
        sc.pl.tsne(dataset, color='timepoint', ax=ax[i], show = False)
    plt.show()

def make_normal_trajectories(datasets):
    """
    Plots trajectories colored by timepoints for all datasets
    :param datasets: List of anndata objects
    :return:
    """
    n_samples = len(datasets)
    fig, ax = plt.subplots(1, n_samples)
    # sc.pl.draw_graph(babos_raw, color=['timepoint', 'Vim', 'Mmp2', 'Cdkn1c', 'Nefl', 'Rbfox3', 'Ncam1'],
    #                  legend_loc='on data')

    for i in range(n_samples):
        dataset = datasets[i]
        sc.pl.draw_graph(dataset, color='timepoint',
                         legend_loc='on data', ax=ax[i], show = False)

    plt.show()


def make_marker_trajectories(datasets, start_list, end_list):
    """
    Plots trajectories colored by marker genes
    :param datasets: List of anndata objects
    :param start_list: List of beginning marker genes
    :param end_list: List of end marker genes
    :return:
    """
    n_samples = len(datasets)
    fig, ax = plt.subplots(2, n_samples)

    for i in range(n_samples):
        dataset = datasets[i]
        sc.pl.draw_graph(dataset, color=start_list[i],
                         legend_loc='on data', ax=ax[0, i], show=False)
        sc.pl.draw_graph(dataset, color=end_list[i],
                         legend_loc='on data', ax=ax[1, i], show=False)

    plt.show()


def main():
    filenames = ['babos_neighbors', 'shie_neighbors', 'fran_neighbors', 'all_neighbors']
    # load harmony integrated datasets checking to see if processes have been run and loading
    # pickle files if they have been to save time

    # runs normalization and integration
    if not path.exists('integrated/all_integrated'):
        print("Making integrated data")
        all_data, all_sets = make_all_raw_datasets(SAMPLES, PATHS, NAMES, META)
        datasets = all_sets

    # runs k nearest neighbors routine
    if not path.exists('neighbors/all_neighbors'):
        print("Loading integrated data")
        with open('integrated/all_integrated', 'rb') as f:
            all_data = pickle.load(f)
        datasets = [all_data[np.equal(all_data.obs['dataset'], name), :] for name in NAMES]
        # make neighbors
        datasets.append(all_data)
        print("Making neighbors")
        for i in range(len(datasets)):
            filename = 'neighbors/' + filenames[i]
            generate_neighbors_analysis(datasets[i], filename)

    # computes trajectories
    if not path.exists('trajectories/all_traj'):
        print("Making trajectories")

        datasets = []
        print("Loading neighbors")
        for i in range(len(filenames)):
            with open('neighbors/' + filenames[i], 'rb') as f:
                data = pickle.load(f)
            datasets.append(data)
        filenames = NAMES + ['all']
        for i in range(len(datasets)):
            dataset = datasets[i]
            sc.tl.paga(dataset, groups='timepoint')
            sc.pl.paga(dataset, threshold=0.0, show=False)
            sc.tl.draw_graph(dataset, init_pos=True)
            with open('trajectories/' + filenames[i] + '_traj', 'wb') as f:
                pickle.dump(dataset, f)

    # computes diffusion pseudotime
    if not path.exists('dpt/babos_dpt'):
        print("Making dpt")
        datasets = []
        filenames = NAMES + ['all']
        print("Loading trajectories")
        for i in range(len(filenames)):
            with open('trajectories/' + filenames[i] + '_traj', 'rb') as f:
                data = pickle.load(f)
            datasets.append(data)
        # set roots for pseudotime analysis
        datasets[0].uns['iroot'] = np.flatnonzero(datasets[0].obs['timepoint'] == 'D0')[0]
        datasets[1].uns['iroot'] = np.flatnonzero(datasets[1].obs['timepoint'] == 'D0')[0]
        datasets[2].uns['iroot'] = np.flatnonzero(datasets[2].obs['timepoint'] == 'D2')[0]

        # compute dpt
        for i in range(3):
            dataset = datasets[i]
            sc.pp.neighbors(dataset, n_neighbors=10, use_rep='X_pca_harmony', method='gauss')
            sc.tl.diffmap(dataset, n_comps=10)
            sc.tl.dpt(dataset, n_dcs=10, n_branchings=1)
            sc.pl.draw_graph(dataset, color='dpt_pseudotime', legend_loc='on data')
            sc.pl.diffmap(dataset, color='dpt_pseudotime')
            with open('dpt/' + NAMES[i] + '_dpt', 'wb') as f:
                pickle.dump(dataset, f)

    # otherwise loads datasets
    else:
        datasets = []
        print("Loading dpt")
        for i in range(len(NAMES)):
            with open('dpt/' + NAMES[i] + '_dpt', 'rb') as f:
                data = pickle.load(f)
            datasets.append(data)

    # sc.pl.draw_graph(datasets[1], color='Vim', legend_loc='on data')

    ### MAKE PLOTS FOR CORRELATION HITS
    babos_hits = ['timepoint', 'Ptgds', 'C1qtnf3', 'Elf3', 'Snhg11', 'Tcf15']
    shie_hits = ['timepoint', 'Ptgds', 'C1qtnf3', 'Elf3', 'Snhg11', 'Dppa3', 'Khdc3', 'Ooep', 'Rhox5',
                 'Rhox6', 'Rhox9', 'Tcf15']
    fran_hits = ['timepoint', 'Elf3', 'Snhg11', 'Dppa3', 'Khdc3', 'Ooep', 'Rhox5',
                 'Rhox6', 'Rhox9']
    sf_hits = ['Dppa3', 'Khdc3', 'Rhox5','Rhox6', 'Rhox9']
    for i in range(2):
        dataset = datasets[i + 1]
        dpt = pd.concat([dataset.obs['dpt_pseudotime'],
                         dataset[:, np.isin(datasets[0].var_names, sf_hits)].to_df()], axis=1)
        dpt = dpt.sort_values(by=['dpt_pseudotime'])
        dpt = dpt.set_index('dpt_pseudotime')
        dpt = dpt.T
        print(dpt)
        fig, ax = plt.subplots(1,1)
        sns.heatmap(dpt, robust=True, xticklabels=False, ax=ax)
        ax.set_xlabel('Diffusion Pseudotime')
        plt.show()

    all_hits = [babos_hits, shie_hits, fran_hits]
    for i in range(len(datasets)):
        dataset = datasets[i]
        dpt = pd.concat([dataset.obs['dpt_pseudotime'],
                               dataset[:, np.isin(datasets[0].var_names, all_hits[i])].to_df()], axis=1)
        dpt = dpt.sort_values(by=['dpt_pseudotime'])
        dpt = dpt.set_index('dpt_pseudotime')
        dpt = dpt.T
        print(dpt)
        fig, ax = plt.subplots(1,1)
        sns.heatmap(dpt, robust=True, xticklabels=False, ax=ax)
        ax.set_xlabel('Diffusion Pseudotime')
        plt.show()

    ### MAKE FIGURE 1 PLOTS
    make_tsne_plots(datasets)
    make_normal_trajectories(datasets)
    make_marker_trajectories(datasets, ['Vim', 'Vim', 'Cd19'], ['Nefl', 'Nanog', 'Nanog'])

    # Define marker genes
    marker_genes_dict1 = {'MEF': ['Vim', 'Mmp2', 'Cdkn1c'],
                         'Neuron': ['Nefl', 'Rbfox3', 'Ncam1']}
    marker_genes_dict2 = {'MEF': ['Vim', 'Mmp2', 'Cdkn1c'],
                         'Stem Cell': ['Zfp42', 'Nanog', 'Dppa5a']}
    marker_genes_dict3 = {'Pre-B': ['Pax5', 'Cd19', 'Cd93'],
                          'Stem Cell': ['Zfp42', 'Nanog', 'Dppa5a']}

    markers = [['Vim', 'Mmp2', 'Cdkn1c', 'Nefl', 'Rbfox3', 'Ncam1'],
               ['Vim', 'Mmp2', 'Cdkn1c', 'Zfp42', 'Nanog', 'Dppa5a'],
               ['Pax5', 'Cd19', 'Cd93', 'Zfp42', 'Nanog', 'Dppa5a']]
    fig, ax = plt.subplots(1, 3)
    for i in range(len(datasets)):
        dataset = datasets[i]
        dpt = pd.concat([dataset.obs['dpt_pseudotime'],
                               dataset[:, np.isin(datasets[0].var_names, markers[i])].to_df()], axis=1)
        dpt = dpt.sort_values(by=['dpt_pseudotime'])
        dpt = dpt.set_index('dpt_pseudotime')
        dpt = dpt.T
        dpt = dpt.reindex(markers[i])
        print(dpt)
        sns.heatmap(dpt, robust=True, xticklabels=False, ax=ax[i])
        ax[i].set_xlabel('Diffusion Pseudotime')
    plt.show()

    generate_dotplots(datasets,
                      [marker_genes_dict1, marker_genes_dict2, marker_genes_dict3])

    ### MAKE FIGURE 3 PLOTS

    # define GSEA hits
    ribosomal_hits = ['Rps18', 'Rpl41', 'Rps28', 'Rps20', 'Rplp0', 'Rplp1', 'Rpl32', 'Rps9',
                      'Rpl13a', 'Rps27a', 'Rpl39', 'Rps4x', 'Rpl18', 'Rps14',
                      'Rps27', 'Rpl13', 'Rps19', 'Rps3']

    xenobiotic_hits = ['Aldh3a1', 'Mgst1', 'Gsto1', 'Mgst3', 'Gstm1']

    canonical_wnt_hits = ['Apoe', 'Col1a1', 'Cthrc1', 'Ddit3', 'Igfbp2', 'Igfbp4', 'Igfbp6']

    gsea_hits = [ribosomal_hits, xenobiotic_hits, canonical_wnt_hits]
    for hits in gsea_hits:
        for i in range(len(datasets)):
            dataset = datasets[i]
            dpt = pd.concat([dataset.obs['dpt_pseudotime'],
                                   dataset[:, np.isin(datasets[0].var_names, hits)].to_df()], axis=1)
            dpt = dpt.sort_values(by=['dpt_pseudotime'])
            dpt = dpt.set_index('dpt_pseudotime')
            dpt = dpt.T
            print(dpt)
            fig, ax = plt.subplots(1,1)
            sns.heatmap(dpt, robust=True, xticklabels=False, ax=ax)
            ax.set_xlabel('Diffusion Pseudotime')
            plt.show()

main()


