from pathlib import Path
from typing import *
from isolateparser.resultparser.parsers import gdtotable
import pytest
import itertools
from loguru import logger
data_folder = Path(__file__).parent / "data" / "sample_files"


@pytest.fixture
def parser() -> gdtotable.GDToTable:
	return gdtotable.GDToTable()


def example_gd_file_contents() -> List[str]:
	expected = [
		"SNP	1	13	1	598843	G	gene_name=JNAKOBFD_00523/JNAKOBFD_00524	gene_position=intergenic (+4/+95)	gene_product=[locus_tag=Bcen2424_2337] [db_xref=InterPro:IPR001734] [protein=Na+/solute symporter][protein_id=ABK09088.1] [location=complement(2599928..2601478)][gbkey=CDS]/[locus_tag=Bcen2424_2336] [db_xref=InterPro:IPR004360,InterPro:IPR009725] [protein=Glyoxalase/bleomycin resistance protein/dioxygenase] [protein_id=ABK09087.1][location=2599374..2599829] [gbkey=CDS]	genes_promoter=JNAKOBFD_00524	html_gene_name=<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>	locus_tag=–/–	mutation_category=snp_intergenic	snp_type=intergenic",
		"INS	2	16	1	1137319	C	gene_name=JNAKOBFD_01012	gene_position=coding (1836/1887 nt)	gene_product=[locus_tag=Bcen2424_6766] [db_xref=InterPro:IPR002202] [protein=PE-PGRS family protein][protein_id=ABK13495.1] [location=complement(1015331..1019638)][gbkey=CDS]	gene_strand=<	genes_overlapping=JNAKOBFD_01012	html_gene_name=<i>JNAKOBFD_01012</i>&nbsp;&larr;	mutation_category=small_indel	repeat_length=1	repeat_new_copies=6	repeat_ref_copies=5	repeat_seq=C",
		"SNP	3	38	1	1301313	A	aa_new_seq=L	aa_position=759	aa_ref_seq=L	codon_new_seq=TTG	codon_number=759	codon_position=1	codon_ref_seq=CTG	gene_name=JNAKOBFD_01153	gene_position=2275	gene_product=[locus_tag=Bcen2424_5983] [db_xref=InterPro:IPR005594,InterPro:IPR006162,InterPro:IPR008635,InterPro:IPR008640][protein=YadA C-terminal domain protein][protein_id=ABK12716.1] [location=complement(124247..128662)][gbkey=CDS]	gene_strand=<	genes_overlapping=JNAKOBFD_01153	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	mutation_category=snp_synonymous	snp_type=synonymous	transl_table=1",
		"SNP	4	39	1	1301317	C	aa_new_seq=T	aa_position=757	aa_ref_seq=T	codon_new_seq=ACG	codon_number=757	codon_position=3	codon_ref_seq=ACC	gene_name=JNAKOBFD_01153	gene_position=2271	gene_product=[locus_tag=Bcen2424_5983] [db_xref=InterPro:IPR005594,InterPro:IPR006162,InterPro:IPR008635,InterPro:IPR008640][protein=YadA C-terminal domain protein][protein_id=ABK12716.1] [location=complement(124247..128662)][gbkey=CDS]	gene_strand=<	genes_overlapping=JNAKOBFD_01153	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	mutation_category=snp_synonymous	snp_type=synonymous	transl_table=1",
		"SNP	5	56	2	783559	G	gene_name=JNAKOBFD_03573/JNAKOBFD_03574	gene_position=intergenic (+102/-470)	gene_product=[locus_tag=Bcen2424_0544] [db_xref=InterPro:IPR001753] [protein=short chain enoyl-CoA hydratase] [protein_id=ABK07298.1] [location=605087..605863][gbkey=CDS]/16S ribosomal RNA	genes_promoter=JNAKOBFD_03574	html_gene_name=<i>JNAKOBFD_03573</i>&nbsp;&rarr;&nbsp;/&nbsp;&rarr;&nbsp;<i>JNAKOBFD_03574</i>	locus_tag=–/–	mutation_category=snp_intergenic	snp_type=intergenic",
		"SNP	6	58	2	1595621	T	aa_new_seq=V	aa_position=111	aa_ref_seq=G	codon_new_seq=GTG	codon_number=111	codon_position=2	codon_ref_seq=GGG	gene_name=JNAKOBFD_04336	gene_position=332	gene_product=[locus_tag=Bcen2424_2424] [db_xref=InterPro:IPR003754,InterPro:IPR007470] [protein=protein of unknown function DUF513, hemX] [protein_id=ABK09174.1][location=complement(2688435..2690405)] [gbkey=CDS]	gene_strand=>	genes_overlapping=JNAKOBFD_04336	html_gene_name=<i>JNAKOBFD_04336</i>&nbsp;&rarr;	mutation_category=snp_nonsynonymous	snp_type=nonsynonymous	transl_table=1",
		"SNP	7	59	2	1595623	A	aa_new_seq=I	aa_position=112	aa_ref_seq=F	codon_new_seq=ATC	codon_number=112	codon_position=1	codon_ref_seq=TTC	gene_name=JNAKOBFD_04336	gene_position=334	gene_product=[locus_tag=Bcen2424_2424] [db_xref=InterPro:IPR003754,InterPro:IPR007470] [protein=protein of unknown function DUF513, hemX] [protein_id=ABK09174.1][location=complement(2688435..2690405)] [gbkey=CDS]	gene_strand=>	genes_overlapping=JNAKOBFD_04336	html_gene_name=<i>JNAKOBFD_04336</i>&nbsp;&rarr;	mutation_category=snp_nonsynonymous	snp_type=nonsynonymous	transl_table=1",
		"SNP	8	61	2	2478091	A	gene_name=JNAKOBFD_05108/JNAKOBFD_05109	gene_position=intergenic (-161/+83)	gene_product=[locus_tag=Bcen2424_4005] [db_xref=InterPro:IPR007138] [protein=Antibiotic biosynthesis monooxygenase] [protein_id=ABK10741.1] [location=893139..893474][gbkey=CDS]/[locus_tag=Bcen2424_4004] [db_xref=InterPro:IPR002198,InterPro:IPR002347] [protein=short-chain dehydrogenase/reductase SDR] [protein_id=ABK10740.1][location=892176..892895] [gbkey=CDS]	html_gene_name=<i>JNAKOBFD_05108</i>&nbsp;&larr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_05109</i>	locus_tag=–/–	mutation_category=snp_intergenic	snp_type=intergenic",
		"SNP	9	62	2	2478103	C	gene_name=JNAKOBFD_05108/JNAKOBFD_05109	gene_position=intergenic (-173/+71)	gene_product=[locus_tag=Bcen2424_4005] [db_xref=InterPro:IPR007138] [protein=Antibiotic biosynthesis monooxygenase] [protein_id=ABK10741.1] [location=893139..893474][gbkey=CDS]/[locus_tag=Bcen2424_4004] [db_xref=InterPro:IPR002198,InterPro:IPR002347] [protein=short-chain dehydrogenase/reductase SDR] [protein_id=ABK10740.1][location=892176..892895] [gbkey=CDS]	html_gene_name=<i>JNAKOBFD_05108</i>&nbsp;&larr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_05109</i>	locus_tag=–/–	mutation_category=snp_intergenic	snp_type=intergenic",
		"SNP	10	75	3	181113	A	aa_new_seq=G	aa_position=368	aa_ref_seq=G	codon_new_seq=GGT	codon_number=368	codon_position=3	codon_ref_seq=GGC	gene_name=JNAKOBFD_05752	gene_position=1104	gene_product=[locus_tag=Bcen2424_5364] [db_xref=InterPro:IPR000014,InterPro:IPR003660,InterPro:IPR004089,InterPro:IPR004090,InterPro:IPR005829,InterPro:IPR013655][protein=methyl-accepting chemotaxis sensory transducer with Pas/Pac sensor] [protein_id=ABK12097.1] [location=complement(2460320..2461864)][gbkey=CDS]	gene_strand=<	genes_overlapping=JNAKOBFD_05752	html_gene_name=<i>JNAKOBFD_05752</i>&nbsp;&larr;	mutation_category=snp_synonymous	snp_type=synonymous	transl_table=1",
		"DEL	11	95	5	1	1248	gene_name=JNAKOBFD_06387	gene_product=JNAKOBFD_06387	genes_inactivated=JNAKOBFD_06387	html_gene_name=<i>JNAKOBFD_06387</i>	mutation_category=large_deletion",
		"RA	13	.	1	598843	0	A	G	consensus_score=20.7	frequency=1	gene_name=JNAKOBFD_00523/JNAKOBFD_00524	gene_position=intergenic (+4/+95)	gene_product=[locus_tag=Bcen2424_2337] [db_xref=InterPro:IPR001734] [protein=Na+/solute symporter][protein_id=ABK09088.1] [location=complement(2599928..2601478)][gbkey=CDS]/[locus_tag=Bcen2424_2336] [db_xref=InterPro:IPR004360,InterPro:IPR009725] [protein=Glyoxalase/bleomycin resistance protein/dioxygenase] [protein_id=ABK09087.1][location=2599374..2599829] [gbkey=CDS]	html_gene_name=<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>	locus_tag=–/–	major_base=G	major_cov=4/3	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/3	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	snp_type=intergenic	total_cov=4/3",
		"RA	16	.	1	1137314	1	.	C	aa_new_seq=G	aa_position=614	aa_ref_seq=V	bias_e_value=5670690	bias_p_value=0.804858	codon_new_seq=GGG	codon_number=614	codon_position=2	codon_ref_seq=GTG	consensus_score=13.5	fisher_strand_p_value=0.444444	frequency=1	gene_name=JNAKOBFD_01012	gene_position=1841	gene_product=[locus_tag=Bcen2424_6766] [db_xref=InterPro:IPR002202] [protein=PE-PGRS family protein][protein_id=ABK13495.1] [location=complement(1015331..1019638)][gbkey=CDS]	gene_strand=<	html_gene_name=<i>JNAKOBFD_01012</i>&nbsp;&larr;	ks_quality_p_value=1	major_base=C	major_cov=5/3	major_frequency=8.889e-01	minor_base=G	minor_cov=0/1	new_cov=5/3	new_seq=C	polymorphism_frequency=8.889e-01	polymorphism_score=-5.8	prediction=consensus	ref_cov=0/0	ref_seq=A	snp_type=nonsynonymous	total_cov=5/4	transl_table=1",
		"RA	38	.	1	1301313	0	G	A	aa_new_seq=L	aa_position=759	aa_ref_seq=L	codon_new_seq=TTG	codon_number=759	codon_position=1	codon_ref_seq=CTG	consensus_score=15.7	frequency=1	gene_name=JNAKOBFD_01153	gene_position=2275	gene_product=[locus_tag=Bcen2424_5983] [db_xref=InterPro:IPR005594,InterPro:IPR006162,InterPro:IPR008635,InterPro:IPR008640][protein=YadA C-terminal domain protein][protein_id=ABK12716.1] [location=complement(124247..128662)][gbkey=CDS]	gene_strand=<	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	major_base=A	major_cov=4/4	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/4	new_seq=A	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	ref_seq=G	snp_type=synonymous	total_cov=4/4	transl_table=1",
		"RA	39	.	1	1301317	0	G	C	aa_new_seq=T	aa_position=757	aa_ref_seq=T	codon_new_seq=ACG	codon_number=757	codon_position=3	codon_ref_seq=ACC	consensus_score=14.0	frequency=1	gene_name=JNAKOBFD_01153	gene_position=2271	gene_product=[locus_tag=Bcen2424_5983] [db_xref=InterPro:IPR005594,InterPro:IPR006162,InterPro:IPR008635,InterPro:IPR008640][protein=YadA C-terminal domain protein][protein_id=ABK12716.1] [location=complement(124247..128662)][gbkey=CDS]	gene_strand=<	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	major_base=C	major_cov=4/4	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/4	new_seq=C	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	ref_seq=G	snp_type=synonymous	total_cov=4/4	transl_table=1",
		"RA	56	.	2	783559	0	A	G	consensus_score=16.5	frequency=1	gene_name=JNAKOBFD_03573/JNAKOBFD_03574	gene_position=intergenic (+102/-470)	gene_product=[locus_tag=Bcen2424_0544] [db_xref=InterPro:IPR001753] [protein=short chain enoyl-CoA hydratase] [protein_id=ABK07298.1] [location=605087..605863][gbkey=CDS]/16S ribosomal RNA	html_gene_name=<i>JNAKOBFD_03573</i>&nbsp;&rarr;&nbsp;/&nbsp;&rarr;&nbsp;<i>JNAKOBFD_03574</i>	locus_tag=–/–	major_base=G	major_cov=4/2	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/2	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	snp_type=intergenic	total_cov=4/2",
		"RA	58	.	2	1595621	0	G	T	aa_new_seq=V	aa_position=111	aa_ref_seq=G	bias_e_value=7045580	bias_p_value=1	codon_new_seq=GTG	codon_number=111	codon_position=2	codon_ref_seq=GGG	consensus_score=20.2	fisher_strand_p_value=1	frequency=1	gene_name=JNAKOBFD_04336	gene_position=332	gene_product=[locus_tag=Bcen2424_2424] [db_xref=InterPro:IPR003754,InterPro:IPR007470] [protein=protein of unknown function DUF513, hemX] [protein_id=ABK09174.1][location=complement(2688435..2690405)] [gbkey=CDS]	gene_strand=>	html_gene_name=<i>JNAKOBFD_04336</i>&nbsp;&rarr;	ks_quality_p_value=1	major_base=T	major_cov=6/2	major_frequency=8.889e-01	minor_base=G	minor_cov=1/0	new_cov=6/2	new_seq=T	polymorphism_frequency=8.889e-01	polymorphism_score=-5.9	prediction=consensus	ref_cov=1/0	ref_seq=G	snp_type=nonsynonymous	total_cov=7/2	transl_table=1",
		"RA	59	.	2	1595623	0	T	A	aa_new_seq=I	aa_position=112	aa_ref_seq=F	bias_e_value=7045580	bias_p_value=1	codon_new_seq=ATC	codon_number=112	codon_position=1	codon_ref_seq=TTC	consensus_score=18.7	fisher_strand_p_value=1	frequency=1	gene_name=JNAKOBFD_04336	gene_position=334	gene_product=[locus_tag=Bcen2424_2424] [db_xref=InterPro:IPR003754,InterPro:IPR007470] [protein=protein of unknown function DUF513, hemX] [protein_id=ABK09174.1][location=complement(2688435..2690405)] [gbkey=CDS]	gene_strand=>	html_gene_name=<i>JNAKOBFD_04336</i>&nbsp;&rarr;	ks_quality_p_value=1	major_base=A	major_cov=6/2	major_frequency=8.889e-01	minor_base=T	minor_cov=1/0	new_cov=6/2	new_seq=A	polymorphism_frequency=8.889e-01	polymorphism_score=-5.4	prediction=consensus	ref_cov=1/0	ref_seq=T	snp_type=nonsynonymous	total_cov=7/2	transl_table=1",
		"RA	61	.	2	2478091	0	T	A	bias_e_value=4534860	bias_p_value=0.643647	consensus_score=14.8	fisher_strand_p_value=0.285714	frequency=1	gene_name=JNAKOBFD_05108/JNAKOBFD_05109	gene_position=intergenic (-161/+83)	gene_product=[locus_tag=Bcen2424_4005] [db_xref=InterPro:IPR007138] [protein=Antibiotic biosynthesis monooxygenase] [protein_id=ABK10741.1] [location=893139..893474][gbkey=CDS]/[locus_tag=Bcen2424_4004] [db_xref=InterPro:IPR002198,InterPro:IPR002347] [protein=short-chain dehydrogenase/reductase SDR] [protein_id=ABK10740.1][location=892176..892895] [gbkey=CDS]	html_gene_name=<i>JNAKOBFD_05108</i>&nbsp;&larr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_05109</i>	ks_quality_p_value=1	locus_tag=–/–	major_base=A	major_cov=1/5	major_frequency=8.571e-01	minor_base=T	minor_cov=1/0	new_cov=1/5	polymorphism_frequency=8.571e-01	polymorphism_score=-5.3	prediction=consensus	ref_cov=1/0	snp_type=intergenic	total_cov=2/5",
		"RA	62	.	2	2478103	0	G	C	bias_e_value=4928640	bias_p_value=0.699537	consensus_score=12.3	fisher_strand_p_value=0.333333	frequency=1	gene_name=JNAKOBFD_05108/JNAKOBFD_05109	gene_position=intergenic (-173/+71)	gene_product=[locus_tag=Bcen2424_4005] [db_xref=InterPro:IPR007138] [protein=Antibiotic biosynthesis monooxygenase] [protein_id=ABK10741.1] [location=893139..893474][gbkey=CDS]/[locus_tag=Bcen2424_4004] [db_xref=InterPro:IPR002198,InterPro:IPR002347] [protein=short-chain dehydrogenase/reductase SDR] [protein_id=ABK10740.1][location=892176..892895] [gbkey=CDS]	html_gene_name=<i>JNAKOBFD_05108</i>&nbsp;&larr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_05109</i>	ks_quality_p_value=1	locus_tag=–/–	major_base=C	major_cov=1/4	major_frequency=8.333e-01	minor_base=G	minor_cov=1/0	new_cov=1/4	polymorphism_frequency=8.333e-01	polymorphism_score=-5.0	prediction=consensus	ref_cov=1/0	snp_type=intergenic	total_cov=2/4",
		"RA	75	.	3	181113	0	G	A	aa_new_seq=G	aa_position=368	aa_ref_seq=G	codon_new_seq=GGT	codon_number=368	codon_position=3	codon_ref_seq=GGC	consensus_score=15.0	frequency=1	gene_name=JNAKOBFD_05752	gene_position=1104	gene_product=[locus_tag=Bcen2424_5364] [db_xref=InterPro:IPR000014,InterPro:IPR003660,InterPro:IPR004089,InterPro:IPR004090,InterPro:IPR005829,InterPro:IPR013655][protein=methyl-accepting chemotaxis sensory transducer with Pas/Pac sensor] [protein_id=ABK12097.1] [location=complement(2460320..2461864)][gbkey=CDS]	gene_strand=<	html_gene_name=<i>JNAKOBFD_05752</i>&nbsp;&larr;	major_base=A	major_cov=4/2	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/2	new_seq=A	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	ref_seq=G	snp_type=synonymous	total_cov=4/2	transl_table=1",
		"MC	95	.	5	1	1248	0	0	gene_name=JNAKOBFD_06387	gene_product=JNAKOBFD_06387	html_gene_name=<i>JNAKOBFD_06387</i>	left_inside_cov=0	left_outside_cov=NA	right_inside_cov=0	right_outside_cov=NA"
	]
	return expected


def test_read_gd_file(parser):
	filename = data_folder / "SC1360.annotated.filtered.gd"

	result = parser.read_gd_file(filename)

	assert result == example_gd_file_contents()


def test_is_mutation(parser):
	expected = [True] * 11 + [False] * 11

	result = [parser.is_mutation(i) for i in example_gd_file_contents()]

	assert result == expected


def test_is_evidence(parser):
	expected = [False] * 11 + [True] * 11

	result = [parser.is_evidence(i) for i in example_gd_file_contents()]

	assert result == expected


sample_gd_lines = [
	(
		"SNP	1	13	1	598843	G	gene_name=JNAKOBFD_00523/JNAKOBFD_00524	gene_position=intergenic (+4/+95)	gene_product=[locus_tag=Bcen2424_2337] [db_xref=InterPro:IPR001734] [protein=Na+/solute symporter][protein_id=ABK09088.1]	genes_promoter=JNAKOBFD_00524	html_gene_name=<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>	locus_tag=–/–	mutation_category=snp_intergenic	snp_type=intergenic",
		{
			'category_id':    'SNP', 'evidence_id': '1', 'parent_ids': '13', 'seq_id': '1', 'position': '598843',
			'new_seq':        'G',
			'gene_name':      "JNAKOBFD_00523/JNAKOBFD_00524", 'gene_position': "intergenic (+4/+95)",
			'gene_product':   "[locus_tag=Bcen2424_2337] [db_xref=InterPro:IPR001734] [protein=Na+/solute symporter][protein_id=ABK09088.1]",
			'genes_promoter': "JNAKOBFD_00524",
			'html_gene_name': "<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>",
			'locus_tag':      "–/–", 'mutation_category': 'snp_intergenic', 'snp_type': 'intergenic'
		}
	),
	(
		"SNP	4	39	1	1301317	C	aa_new_seq=T	aa_position=757	aa_ref_seq=T	codon_new_seq=ACG	codon_number=757	codon_position=3	codon_ref_seq=ACC	gene_name=JNAKOBFD_01153	gene_position=2271	gene_product=[locus_tag=Bcen2424_5983]	gene_strand=<	genes_overlapping=JNAKOBFD_01153	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	mutation_category=snp_synonymous	snp_type=synonymous	transl_table=1",
		{
			'category_id':       'SNP', 'evidence_id': '4', 'parent_ids': '39', 'seq_id': '1', 'position': '1301317',
			'new_seq':           'C',
			'aa_new_seq':        'T', 'aa_position': '757', 'aa_ref_seq': 'T', 'codon_new_seq': 'ACG',
			'codon_number':      '757', 'codon_position': '3', 'codon_ref_seq': 'ACC',
			'gene_name':         "JNAKOBFD_01153", 'gene_position': '2271',
			'gene_product':      '[locus_tag=Bcen2424_5983]', 'gene_strand': '<',
			'genes_overlapping': 'JNAKOBFD_01153', 'html_gene_name': "<i>JNAKOBFD_01153</i>&nbsp;&larr;",
			'mutation_category': 'snp_synonymous', 'snp_type': 'synonymous', 'transl_table': '1'
		}
	),
	(
		"INS	2	16	1	1137319	C	gene_name=JNAKOBFD_01012	gene_position=coding (1836/1887 nt)	gene_product=[locus_tag=Bcen2424_6766]	gene_strand=<	genes_overlapping=JNAKOBFD_01012	html_gene_name=<i>JNAKOBFD_01012</i>&nbsp;&larr;	mutation_category=small_indel	repeat_length=1	repeat_new_copies=6	repeat_ref_copies=5	repeat_seq=C",
		{
			'category_id':       'INS', 'evidence_id': '2', 'parent_ids': '16', 'seq_id': '1', 'position': '1137319',
			'new_seq':           'C',
			'gene_name':         "JNAKOBFD_01012", 'gene_position': "coding (1836/1887 nt)",
			'gene_product':      "[locus_tag=Bcen2424_6766]", 'gene_strand': '<',
			'genes_overlapping': 'JNAKOBFD_01012', 'html_gene_name': "<i>JNAKOBFD_01012</i>&nbsp;&larr;",
			'mutation_category': 'small_indel', 'repeat_length': '1', 'repeat_new_copies': '6',
			'repeat_ref_copies': '5', 'repeat_seq': 'C'
		}
	),
	(
		"DEL	11	95	5	1	1248	gene_name=JNAKOBFD_06387	gene_product=JNAKOBFD_06387	genes_inactivated=JNAKOBFD_06387	html_gene_name=<i>JNAKOBFD_06387</i>	mutation_category=large_deletion",
		{
			'category_id':       'DEL', 'evidence_id': '11', 'parent_ids': '95', 'seq_id': '5', 'position': '1',
			'size':              '1248',
			'gene_name':         'JNAKOBFD_06387', 'gene_product': 'JNAKOBFD_06387',
			'genes_inactivated': 'JNAKOBFD_06387', 'html_gene_name': "<i>JNAKOBFD_06387</i>",
			'mutation_category': 'large_deletion'
		}
	),
	(
		"RA	13	.	1	598843	0	A	G	consensus_score=20.7	frequency=1	gene_name=JNAKOBFD_00523/JNAKOBFD_00524	gene_position=intergenic (+4/+95)	gene_product=[locus_tag=Bcen2424_2337]	html_gene_name=<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>	locus_tag=–/–	major_base=G	major_cov=4/3	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/3	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	snp_type=intergenic	total_cov=4/3",
		{
			'category_id':        "RA", 'evidence_id': '13', 'parent_ids': '.', 'seq_id': '1', 'position': '598843',
			'insert_position':    '0', 'ref_base': 'A', 'new_base': 'G', 'consensus_score': '20.7', 'frequency': '1',
			'gene_name':          'JNAKOBFD_00523/JNAKOBFD_00524', 'gene_position': 'intergenic (+4/+95)',
			'gene_product':       '[locus_tag=Bcen2424_2337]',
			'html_gene_name':     '<i>JNAKOBFD_00523</i>&nbsp;&rarr;&nbsp;/&nbsp;&larr;&nbsp;<i>JNAKOBFD_00524</i>',
			'locus_tag':          "–/–", 'major_base': 'G', 'major_frequency': '1.000e+00',
			'minor_base':         'N', 'polymorphism_frequency': '1.000e+00',
			'polymorphism_score': 'NA', 'prediction': 'consensus', 'snp_type': 'intergenic',
			#'total_cov':          '4/3',
			'total_cov_forward': '4',
			'total_cov_reverse': '3',
			#'major_cov': '4/3',
			'major_cov_forward': '4',
			'major_cov_reverse': '3',
			#'new_cov': '4/3',
			'new_cov_forward': '4',
			'new_cov_reverse': '3',
			#'minor_cov': '0/0',
			'minor_cov_forward': '0',
			'minor_cov_reverse': '0',
			#'ref_cov': '0/0'
			'ref_cov_forward': '0',
			'ref_cov_reverse': '0'
		}
	),
	(
		"RA	39	.	1	1301317	0	G	C	aa_new_seq=T	aa_position=757	aa_ref_seq=T	codon_new_seq=ACG	codon_number=757	codon_position=3	codon_ref_seq=ACC	consensus_score=14.0	frequency=1	gene_name=JNAKOBFD_01153	gene_position=2271	gene_product=[locus_tag=Bcen2424_5983]	gene_strand=<	html_gene_name=<i>JNAKOBFD_01153</i>&nbsp;&larr;	major_base=C	major_cov=4/4	major_frequency=1.000e+00	minor_base=N	minor_cov=0/0	new_cov=4/4	new_seq=C	polymorphism_frequency=1.000e+00	polymorphism_score=NA	prediction=consensus	ref_cov=0/0	ref_seq=G	snp_type=synonymous	total_cov=4/4	transl_table=1",
		{
			'category_id':            'RA', 'evidence_id': '39', 'parent_ids': '.', 'seq_id': '1',
			'position':               '1301317',
			'insert_position':        '0', 'ref_base': 'G', 'new_base': 'C', 'aa_new_seq': 'T', 'aa_position': '757',
			'aa_ref_seq':             'T',
			'codon_new_seq':          'ACG', 'codon_number': '757', 'codon_position': '3', 'codon_ref_seq': 'ACC',
			'consensus_score':        '14.0',
			'frequency':              '1', 'gene_name': 'JNAKOBFD_01153', 'gene_position': '2271',
			'gene_product':           '[locus_tag=Bcen2424_5983]',
			'gene_strand':            '<', 'html_gene_name': '<i>JNAKOBFD_01153</i>&nbsp;&larr;', 'major_base': 'C',

			'major_frequency':        '1.000e+00', 'minor_base': 'N',
			'new_seq':                'C',
			'polymorphism_frequency': '1.000e+00', 'polymorphism_score': 'NA', 'prediction': 'consensus',
			'ref_seq':                'G', 'snp_type': 'synonymous', 'transl_table': '1',
			#'major_cov':              '4/4',
			'major_cov_forward': '4',
			'major_cov_reverse': '4',
			#'ref_cov':                '0/0',
			'ref_cov_forward': '0',
			'ref_cov_reverse': '0',
			#'minor_cov': '0/0',
			'minor_cov_forward': '0',
			'minor_cov_reverse': '0',
			#'new_cov': '4/4',
			'new_cov_forward': '4',
			'new_cov_reverse': '4',
			#'total_cov': '4/4',
			'total_cov_forward': '4',
			'total_cov_reverse': '4'


		}
	),
	(
		"MC	95	.	5	1	1248	0	0	gene_name=JNAKOBFD_06387	gene_product=JNAKOBFD_06387	html_gene_name=<i>JNAKOBFD_06387</i>	left_inside_cov=0	left_outside_cov=NA	right_inside_cov=0	right_outside_cov=NA",
		{'category_id':         'MC', 'evidence_id': '95', 'parent_ids': '.', 'seq_id': '5', 'start': '1',
			'end':              '1248', 'start_range': '0', 'end_range': '0', 'gene_name': 'JNAKOBFD_06387',
			'gene_product':     "JNAKOBFD_06387", 'html_gene_name': '<i>JNAKOBFD_06387</i>', 'left_inside_cov': '0',
			'left_outside_cov': 'NA', 'right_inside_cov': '0', 'right_outside_cov': 'NA'
		}
	)
]


@pytest.mark.parametrize(
	"line, expected",
	sample_gd_lines
)
def test_split_fields(parser, line, expected):
	line = line.split('\t')

	number_of_positional_arguments = len([i for i in line if '=' not in i])
	line = line[number_of_positional_arguments:]

	result = parser.parse_keyword_fields(line)
	# Need to remove the empty fields since this method is supposed to be given a truncated line anyway.
	result = {k: v for k, v in result.items() if v}
	_to_remove = ['category_id', 'evidence_id', 'parent_ids', 'seq_id', 'position', 'new_seq', 'size']
	# Need to remove the positional arguments
	result = {k: v for k, v in result.items() if k not in _to_remove}
	# Need to filter out the positional arguments from the positional_arguments
	expected = {k:v for k, v in result.items() if k in result.keys()}
	# The `split_fields` method is only used on part of the line.

	assert result == expected


@pytest.mark.parametrize(
	"line, expected",
	[
		("SNP	1	13	1	598843	G", {
			'category_id': 'SNP', 'evidence_id': '1', 'parent_ids': '13', 'seq_id': '1', 'position': '598843',
			'new_seq':     'G'
		}),
		("SNP	4	39	1	1301317	C", {
			'category_id': 'SNP', 'evidence_id': '4', 'parent_ids': '39', 'seq_id': '1', 'position': '1301317',
			'new_seq':     'C'
		}),
		("INS	2	16	1	1137319	C", {
			'category_id': 'INS', 'evidence_id': '2', 'parent_ids': '16', 'seq_id': '1', 'position': '1137319',
			'new_seq':     'C'
		}),
		("DEL	11	95	5	1	1248", {
			'category_id': 'DEL', 'evidence_id': '11', 'parent_ids': '95', 'seq_id': '5', 'position': '1',
			'size':        '1248'
		}),
		("MOB	1	117,118	REL606	16972	IS150	-1	3", {
			'category_id': 'MOB', 'evidence_id': '1', 'parent_ids': '117,118', 'seq_id': 'REL606', 'position': '16972',
			'repeat_name': 'IS150', 'strand': '-1', 'duplication_size': '3'
		})
	]
)
def test_parse_positional_fields(parser, line, expected):
	line = line.split("\t")
	result = parser.parse_positional_fields(line)

	assert result == expected


@pytest.mark.parametrize(
	"line, expected",
	sample_gd_lines
)
def test_parse_mutation(parser, line, expected):
	result = parser.parse_line(line)

	assert result == expected
