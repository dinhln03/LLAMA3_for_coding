import pandas as pd

kraken_rank_dictionary = {
    'P': 'phylum',
    'C': 'class',
    'O': 'order',
    'F': 'family',
    'G': 'genus',
    'S': 'species'
}

greengenes_rank_dict = {
    'k__': 'kingdom',
    'p__': 'phylum',
    'c__': 'class',
    'o__': 'order',
    'f__': 'family',
    'g__': 'genus',
    's__': 'species'
}

kraken_columns = ['PERCENTAGE', 'lca_read_count', 'read_count', 'rank',
                  '@@TAXID', 'TAXNAME']


def kraken2_transformer(all_rank_summary, output_rank_summaries, ranks):
    # TODO finsih docs
    """Converts a summary of all ranks from kraken into rank-wise profiles
    similar to the CAMI-SIM output

    Parameters
    ----------
    all_rank_summary
    output_rank_summaries
    ranks

    Returns
    -------

    """
    # TODO COULD be split into two format functions: one to reformat,
    #  and one to split on rank
    # TODO give error for invalid rank value
    all_ranks = pd.read_csv(all_rank_summary, sep='\t')
    all_ranks.columns = kraken_columns
    # TODO for kraken is it okay to just take the first part (drop the number)
    all_ranks['rank'] = all_ranks['rank'].str[0]
    all_ranks = all_ranks.loc[all_ranks['rank'].isin(kraken_rank_dictionary)]
    all_ranks['RANK'] = [kraken_rank_dictionary[key] for key in
                         all_ranks['rank']]
    keep_cols = ['@@TAXID', 'RANK', 'TAXNAME', 'PERCENTAGE']
    for output_, rank in zip(output_rank_summaries, ranks):
        sub_df = all_ranks.loc[all_ranks['RANK'] == rank]
        sub_df_matching = sub_df[keep_cols]
        sub_df_matching.to_csv(output_, sep='\t', index=False)


def metaphlan2_transformer(all_rank_summary, output_rank_summaries, ranks):
    all_ranks = pd.read_csv(all_rank_summary, sep='\t', skiprows=3)
    def last_entry(x): return x.split('|')[-1]
    all_ranks['last_clade'] = all_ranks['#clade_name'].map(last_entry)
    all_ranks['@@TAXID'] = all_ranks['NCBI_tax_id'].map(last_entry)
    all_ranks['RANK'] = all_ranks['last_clade'].map(
        lambda x: greengenes_rank_dict[x[:3]])
    all_ranks['TAXNAME'] = all_ranks['last_clade'].map(lambda x: x[3:])
    all_ranks['PERCENTAGE'] = all_ranks['relative_abundance']

    keep_cols = ['@@TAXID', 'RANK', 'TAXNAME', 'PERCENTAGE']
    for output_, rank in zip(output_rank_summaries, ranks):
        sub_df = all_ranks.loc[all_ranks['RANK'] == rank]
        sub_df_matching = sub_df[keep_cols]
        sub_df_matching.to_csv(output_, sep='\t', index=False)
