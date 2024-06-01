# -*- coding: utf-8 -*-
"""Functions for retrieving raw and processed run data"""

from datetime import datetime
from html_table_parser import HTMLTableParser
from utils import convert_numbers
import commands
import vcf


def laneHTML(run, path):
    """Retrieve data from the lane.html page, the data is the general run data and date per lane"""
    try:
        lane_dict = {}
        data_run = {}
        epoch = datetime.utcfromtimestamp(0)

        dict_run = {
            'Cluster_Raw': {'column': 'Clusters (Raw)'},
            'Cluster_PF': {'column': 'Clusters(PF)'},
            'Yield_Mbases': {'column': 'Yield (MBases)'}
        }

        dict_lane = {
            'Lane': {'column': 'Lane'},
            'PF_Clusters': {'column': 'PF Clusters'},
            'PCT_of_lane': {'column': '% of the lane'},
            'PCT_perfect_barcode': {'column': '% Perfect barcode'},
            'PCT_one_mismatch_barcode': {'column': '% One mismatch barcode'},
            'Yield_Mbases': {'column': 'Yield (Mbases)'},
            'PCT_PF_Clusters': {'column': '% PF Clusters'},
            'PCT_Q30_bases': {'column': '% = Q30 bases'},
            'Mean_Quality_Score': {'column': 'Mean Quality Score'}
        }

        date = run.split('_')[0]
        date = '20' + date[0:2] + '-' + date[2:4] + '-' + date[4:6]
        d = datetime.strptime(date, '%Y-%m-%d')
        as_date = (d-epoch).days

        lanehtml = commands.getoutput('find {path}/Data/Intensities/BaseCalls/Reports/html/*/all/all/all/ -iname \'lane.html\''.format(
            path=str(path)
            ))

        with open(lanehtml, 'r') as lane:
            html = lane.read()
            tableParser = HTMLTableParser()
            tableParser.feed(html)
            tables = tableParser.tables                         # tables[1]==run tables[2]==lane

            header_run = tables[1][0]
            header_lane = tables[2][0]

            for col in dict_run:
                dict_run[col]['index'] = header_run.index(dict_run[col]['column'])

            for col in dict_lane:
                dict_lane[col]['index'] = header_lane.index(dict_lane[col]['column'])

            stats_run = tables[1][1]
            stats_run = [convert_numbers(item.replace(',', '')) for item in stats_run]
            for col in dict_run:
                stat = stats_run[dict_run[col]['index']]
                stat = int(stat)
                data_run[col] = stat

            data_run['Date'] = date
            data_run['asDate'] = as_date

            for lane in tables[2][1:]:
                data_lane = {}
                lane = [convert_numbers(item.replace(',', '')) for item in lane]
                lane_num = lane[header_lane.index('Lane')]
                for col in dict_lane:
                    stat = lane[dict_lane[col]['index']]
                    data_lane[col] = stat

                lane_dict[lane_num] = data_lane

        return data_run, lane_dict

    except Exception, e:
        print(e)


def laneBarcodeHTML(path):
    """Retrieve data from the laneBarcode.html page, the data is per barcode/sample per lane"""
    try:
        samples_dict = {}

        dict_samples = {
            'Lane': {'column': 'Lane'},
            'Project': {'column': 'Project'},
            'Sample_name': {'column': 'Sample'},
            'Barcode_sequence': {'column': 'Barcode sequence'},
            'PF_Clusters': {'column': 'PF Clusters'},
            'PCT_of_lane': {'column': '% of the lane'},
            'PCT_perfect_barcode': {'column': '% Perfect barcode'},
            'PCT_one_mismatch_barcode': {'column': '% One mismatch barcode'},
            'Yield_Mbases': {'column': 'Yield (Mbases)'},
            'PCT_PF_Clusters': {'column': '% PF Clusters'},
            'PCT_Q30_bases': {'column': '% = Q30 bases'},
            'Mean_Quality_Score': {'column': 'Mean Quality Score'}
        }

        samplehtml = commands.getoutput('find {path}/Data/Intensities/BaseCalls/Reports/html/*/all/all/all/ -iname \'laneBarcode.html\''.format(
            path=str(path)
            ))

        with open(samplehtml, 'r') as sample:
            html = sample.read()
            tableParser = HTMLTableParser()
            tableParser.feed(html)
            tables = tableParser.tables                         # tables[1]==run tables[2]==sample

            header_samplehtml = tables[2][0]

            for col in dict_samples:
                dict_samples[col]['index'] = header_samplehtml.index(dict_samples[col]['column'])

            for sample_lane in tables[2][1:]:
                data_sample_lane = {}
                if sample_lane[header_samplehtml.index('Project')].upper() != 'DEFAULT':
                    stats = [convert_numbers(item.replace(',', '')) for item in sample_lane]

                    lane = stats[header_samplehtml.index('Lane')]
                    sample = stats[header_samplehtml.index('Sample')]
                    lane_sample = str(lane) + '--' + str(sample)

                    for col in dict_samples:
                        stat = stats[dict_samples[col]['index']]
                        data_sample_lane[col] = stat

                    samples_dict[lane_sample] = data_sample_lane

        return samples_dict

    except Exception, e:
        print(e)


def vcf_file(path):
    """Retrieve data from a vcf file, for each sample the number of variants,
    homo- and heterozygous, number of dbSNP variants and PASS variants is determained
    """
    try:
        dic_samples = {}
        file_vcf = commands.getoutput('find {path}/ -maxdepth 1 -iname \'*.vcf\''.format(
            path=str(path)
            ))
        with open(file_vcf, 'r') as vcffile:
            vcf_file = vcf.Reader(vcffile)
            list_samples = vcf_file.samples
            for sample in list_samples:
                dic_samples[sample] = [0, 0, 0]
            for variant in vcf_file:
                samples = []
                if 'DB'in variant.INFO:
                    DB = 1
                else:
                    DB = 0

                if not variant.FILTER:
                    PASS = 1
                else:
                    PASS = 0

                if variant.num_het != 0:
                    het_samples = variant.get_hets()
                    samples = [item.sample for item in het_samples]
                if variant.num_hom_alt != 0:
                    hom_samples = [item.sample for item in variant.get_hom_alts()]
                    samples.extend(hom_samples)

                for sample in samples:
                    stats = dic_samples[sample]
                    stats[0] += 1
                    stats[1] += DB
                    stats[2] += PASS
                    dic_samples[sample] = stats

        return dic_samples
        # dic_samples[sample name] = [number of variant, Percentage dbSNP variants from total, Percentage PASS variants from total]

    except Exception, e:
        print(e)


def runstat_file(path):
    """Retrieve data from the runstats file,
    for each sample the percentage duplication is retrieved
    """
    try:
        sample_dup = {}
        sample_total_reads = {}
        sample_mapped_percentage = {}
        runstats_file = commands.getoutput('find {path}/ -iname \'run_stats.txt\''.format(
            path=str(path)
            ))

        with open(runstats_file, 'r') as runstats:
            run_stats = runstats.read()
            run_stats = run_stats.split('working on ')

            for sample in run_stats[1:]:
                stats = sample.split('\n')

                sample_name = stats[0].split('/')[-1]
                sample_name = sample_name.replace('.flagstat...', '')

                dup = 0
                total_reads = 0
                mapped_percentage = 0
                for x in stats:
                    if 'total (QC-passed reads + QC-failed reads)' in x:
                        total_reads = int(x.strip().split()[0])
                    elif 'mapped (' in x:
                        mapped_percentage = float(x.split('(')[-1].split('%')[0])
                    elif '%duplication' in x:
                        dup = float(x.split('%')[0].strip('\t').strip())
                        dup = float('{0:.2f}'.format(dup))

                sample_dup[sample_name] = dup
                sample_total_reads[sample_name] = total_reads
                sample_mapped_percentage[sample_name] = mapped_percentage

        return sample_dup, sample_total_reads, sample_mapped_percentage
        # sample_dup[sample name] = duplication

    except Exception, e:
        print(e)


def HSMetrics(path):
    """Retrieve data from the HSMetrics_summary.transposed file,
    from this file all the data is transferred to a dictionary
    """
    try:
        sample_stats = {}
        QCStats_file = commands.getoutput('find {path} -iname \'HSMetrics_summary.txt\''.format(
            path=str(path)
            ))

        dict_columns = {
                'Sample_name': {'column': 'Sample'},
                'Total_reads': {'column': 'TOTAL_READS'},
                'PF_reads': {'column': 'PF_READS'},
                'PF_unique_reads': {'column': 'PF_UNIQUE_READS'},
                'PCT_PF_reads': {'column': 'PCT_PF_READS'},
                'PCT_PF_UQ_reads': {'column': 'PCT_PF_UQ_READS'},
                'PF_UQ_reads_aligned': {'column': 'PF_UQ_READS_ALIGNED'},
                'PCT_PF_UQ_reads_aligned': {'column': 'PCT_PF_UQ_READS_ALIGNED'},
                'PF_UQ_bases_aligned': {'column': 'PF_UQ_BASES_ALIGNED'},
                'On_bait_bases': {'column': 'ON_BAIT_BASES'},
                'Near_bait_bases': {'column': 'NEAR_BAIT_BASES'},
                'Off_bait_bases': {'column': 'OFF_BAIT_BASES'},
                'On_target_bases': {'column': 'ON_TARGET_BASES'},
                'PCT_selected_bases': {'column': 'PCT_SELECTED_BASES'},
                'PCT_off_bait': {'column': 'PCT_OFF_BAIT'},
                'On_bait_vs_selected': {'column': 'ON_BAIT_VS_SELECTED'},
                'Mean_bait_coverage': {'column': 'MEAN_BAIT_COVERAGE'},
                'Mean_target_coverage': {'column': 'MEAN_TARGET_COVERAGE'},
                'PCT_usable_bases_on_bait': {'column': 'PCT_USABLE_BASES_ON_BAIT'},
                'PCT_usable_bases_on_target': {'column': 'PCT_USABLE_BASES_ON_TARGET'},
                'Fold_enrichment': {'column': 'FOLD_ENRICHMENT'},
                'Zero_CVG_targets_PCT': {'column': 'ZERO_CVG_TARGETS_PCT'},
                'Fold_80_base_penalty': {'column': 'FOLD_80_BASE_PENALTY'},
                'PCT_target_bases_2X': {'column': 'PCT_TARGET_BASES_2X'},
                'PCT_target_bases_10X': {'column': 'PCT_TARGET_BASES_10X'},
                'PCT_target_bases_20X': {'column': 'PCT_TARGET_BASES_20X'},
                'PCT_target_bases_30X': {'column': 'PCT_TARGET_BASES_30X'},
                'PCT_target_bases_40X': {'column': 'PCT_TARGET_BASES_40X'},
                'PCT_target_bases_50X': {'column': 'PCT_TARGET_BASES_50X'},
                'PCT_target_bases_100X': {'column': 'PCT_TARGET_BASES_100X'},
                'HS_library_size': {'column': 'HS_LIBRARY_SIZE'},
                'HS_penalty_10X': {'column': 'HS_PENALTY_10X'},
                'HS_penalty_20X': {'column': 'HS_PENALTY_20X'},
                'HS_penalty_30X': {'column': 'HS_PENALTY_30X'},
                'HS_penalty_40X': {'column': 'HS_PENALTY_40X'},
                'HS_penalty_50X': {'column': 'HS_PENALTY_50X'},
                'HS_penalty_100X': {'column': 'HS_PENALTY_100X'},
                'AT_dropout': {'column': 'AT_DROPOUT'},
                'GC_dropout': {'column': 'GC_DROPOUT'},
                'Bait_name': {'column': 'BAIT_SET'},
                'Genome_Size': {'column': 'GENOME_SIZE'},
                'Bait_territory': {'column': 'BAIT_TERRITORY'},
                'Target_territory': {'column': 'TARGET_TERRITORY'},
                'Bait_design_efficiency': {'column': 'BAIT_DESIGN_EFFICIENCY'}
        }

        col_to_pct = ['Bait_design_efficiency', 'PCT_PF_reads', 'PCT_PF_UQ_reads',
                      'PCT_PF_UQ_reads_aligned', 'PCT_selected_bases',
                      'PCT_off_bait', 'On_bait_vs_selected', 'PCT_usable_bases_on_bait',
                      'PCT_usable_bases_on_target', 'Zero_CVG_targets_PCT', 'PCT_target_bases_2X',
                      'PCT_target_bases_10X', 'PCT_target_bases_20X', 'PCT_target_bases_30X',
                      'PCT_target_bases_40X', 'PCT_target_bases_50X', 'PCT_target_bases_100X'
                      ]
        col_format = ['Mean_bait_coverage', 'Mean_target_coverage', 'Fold_enrichment',
                      'Fold_80_base_penalty', 'HS_penalty_10X', 'HS_penalty_20X',
                      'HS_penalty_30X', 'HS_penalty_40X', 'HS_penalty_50X',
                      'HS_penalty_100X', 'AT_dropout', 'GC_dropout'
                      ]

        with open(QCStats_file, 'r') as QCStats:
            sample = []
            qc_stats = QCStats.read().split('\n')
            for line in qc_stats:
                l = line.split('\t')
                sample.append(l)

            qc_table = [list(i) for i in map(None, *sample)]
            qc_table[0][0] = 'Sample'
            table_header = qc_table[0][:-1]
            table_header = [item.replace(' ', '_') for item in table_header]

            for col in dict_columns:
                dict_columns[col]['index'] = table_header.index(dict_columns[col]['column'])

            for stats in qc_table[1:]:
                data_dict = {}
                stats = stats[:-1]                                              # there is a None at the end of each line
                sample_name = stats[table_header.index('Sample')]
                sample_name = sample_name.replace('_dedup', '')
                for col in dict_columns:
                    if col == 'Percentage_reads_mapped':
                        stat = stats[dict_columns[col]['index']]
                        stat = float(stat.strip('%'))
                        data_dict[col] = stat
                    elif col in col_to_pct:
                        stat = stats[dict_columns[col]['index']]
                        stat = float(stat)*100
                        stat = float('{0:.2f}'.format(stat))
                        data_dict[col] = stat
                    elif col in col_format:
                        stat = float(stats[dict_columns[col]['index']])
                        stat = float('{0:.2f}'.format(stat))
                        data_dict[col] = stat
                    elif col == 'Sample_name':
                        data_dict[col] = sample_name
                    else:
                        data_dict[col] = stats[dict_columns[col]['index']]

                sample_stats[sample_name] = data_dict

        return sample_stats

    except Exception, e:
        print(e)
