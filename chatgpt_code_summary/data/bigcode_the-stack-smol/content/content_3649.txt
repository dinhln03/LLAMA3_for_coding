import sys, gzip, logging

from .in_util import TimeReport, detectFileChrom, extendFileList, dumpReader
#========================================
# Schema for AStorage
#========================================

_TRASCRIPT_PROPERTIES = [
    {"name": "Ensembl_geneid",          "tp": "str", "opt": "repeat"},
    {"name": "Ensembl_transcriptid",    "tp": "str", "opt": "repeat"},
    {"name": "Ensembl_proteinid",       "tp": "str", "opt": "repeat"},
    {"name": "refcodon",                "tp": "str", "opt": "repeat"},
    {"name": "codonpos",                "tp": "str", "opt": "repeat"},
    {"name": "FATHMM_score",            "tp": "num"},
    {"name": "FATHMM_pred",             "tp": "str", "opt": "dict"},
    {"name": "GENCODE_basic",           "tp": "str"},
    {"name": "HGVSc_ANNOVAR",           "tp": "str"},
    {"name": "HGVSp_ANNOVAR",           "tp": "str"},
    {"name": "HGVSc_snpEff",            "tp": "str"},
    {"name": "HGVSp_snpEff",            "tp": "str"},
    {"name": "MPC_score",               "tp": "num"},
    {"name": "MutationTaster_score",    "tp": "num"},
    {"name": "MutationAssessor_pred",   "tp": "str", "opt": "dict"},
    {"name": "Polyphen2_HDIV_score",    "tp": "num"},
    {"name": "Polyphen2_HDIV_pred",     "tp": "str", "opt": "dict"},
    {"name": "Polyphen2_HVAR_score",    "tp": "num"},
    {"name": "Polyphen2_HVAR_pred",     "tp": "str", "opt": "dict"},
    {"name": "SIFT_score",              "tp": "num"},
    {"name": "SIFT_pred",               "tp": "str", "opt": "dict"},
    {"name": "SIFT4G_score",            "tp": "num"},
    {"name": "SIFT4G_pred",             "tp": "str", "opt": "dict"},
    {"name": "Uniprot_acc",              "tp": "str"}
]

#===============================================
_FACETS_PROPERTIES = [
    {"name": "MetaLR_score",                "tp": "num"},
    {"name": "MetaLR_rankscore",            "tp": "num"},
    {"name": "MetaLR_pred", "opt": "dict",  "tp": "str"},
    {"name": "MutPred_score",               "tp": "str"},
    {"name": "MutPred_rankscore",           "tp": "num"},
    {"name": "MutPred_protID",              "tp": "str"},
    {"name": "MutPred_AAchange",            "tp": "str"},
    {"name": "MutPred_Top5features",        "tp": "str"},
    {"name": "MPC_rankscore",               "tp": "num"},
    {"name": "PrimateAI_score",             "tp": "num"},
    {"name": "PrimateAI_rankscore",         "tp": "num"},
    {"name": "REVEL_score",                 "tp": "num"},
    {"name": "SIFT4G_converted_rankscore",  "tp": "num"},
    {
        "name": "transcripts", "tp": "list",
        "item": {
            "tp": "dict", "items": _TRASCRIPT_PROPERTIES
        }
    }
]

#===============================================
_VARIANT_PROPERTIES = [
    {"name": "ALT", "tp": "str", "opt": "gene"},
    {"name": "REF", "tp": "str", "opt": "gene"},
    {"name": "CADD_raw",                        "tp": "num"},
    {"name": "CADD_phred",                      "tp": "num"},
    {"name": "DANN_score",                      "tp": "num"},
    {"name": "DANN_rankscore",                  "tp": "num"},
    {"name": "Eigen_raw_coding",                "tp": "num"},
    {"name": "Eigen_raw_coding_rankscore",      "tp": "num"},
    {"name": "Eigen_phred_coding",              "tp": "num"},
    {"name": "Eigen_PC_raw_coding",             "tp": "num"},
    {"name": "Eigen_PC_raw_coding_rankscore",   "tp": "num"},
    {"name": "Eigen_PC_phred_coding",           "tp": "num"},
    {"name": "GTEx_V7_gene",                "tp": "str", "opt": "repeat"},
    {"name": "GTEx_V7_tissue",              "tp": "str"},
    {"name": "MutationTaster_score",        "tp": "str"},
    {"name": "MutationTaster_pred",         "tp": "str"},
    {"name": "PrimateAI_pred",              "tp": "str", "opt": "dict"},
    {"name": "Geuvadis_eQTL_target_gene",   "tp": "str"},
    {
        "name": "facets",
        "tp": "list",
        "item": {
            "tp": "dict",
            "items": _FACETS_PROPERTIES
        }
    }
]

#===============================================
SCHEMA_DBNSFP_4 = {
    "name": "DBNSFP",
    "key": "hg38",
    "io": {
        "block-type": "page-cluster",
        "max-var-count": 50
    },
    "filter-list": {"ref": "REF", "alt": "ALT"},
    "top": {
        "tp": "list",
        "item": {
            "tp": "dict",
            "items": _VARIANT_PROPERTIES
        }
    }
}

#========================================
# Ingest logic
#========================================
VARIANT_TAB = [
    ["REF",                             str],
    ["ALT",                             str],
    ["MutationTaster_score",            str],
    ["MutationTaster_pred",             str],
    ["PrimateAI_pred",                  str],
    ["CADD_raw",                        float],
    ["CADD_phred",                      float],
    ["DANN_score",                      float],
    ["DANN_rankscore",                  float],
    ["Eigen_raw_coding",                float],
    ["Eigen_raw_coding_rankscore",      float],
    ["Eigen_phred_coding",              float],
    ["Eigen_PC_raw_coding",             float],
    ["Eigen_PC_raw_coding_rankscore",   float],
    ["Eigen_PC_phred_coding",           float],
    ["GTEx_V7_gene",                    str],
    ["GTEx_V7_tissue",                  str],
    ["Geuvadis_eQTL_target_gene",       str]
]

#========================================
FACET_TAB = [
    ["refcodon",                    str],
    ["codonpos",                    str],
    ["SIFT4G_converted_rankscore",  float],
    ["MetaLR_score",                float],
    ["MetaLR_rankscore",            float],
    ["MetaLR_pred",                 str],
    ["REVEL_score",                 float],
    ["MutPred_score",               str],
    ["MutPred_rankscore",           float],
    ["MutPred_protID",              str],
    ["MutPred_AAchange",            str],
    ["MutPred_Top5features",        str],
    ["MPC_rankscore",               float],
    ["PrimateAI_score",             float],
    ["PrimateAI_rankscore",         float]
]

#========================================
TRANSCRIPT_TAB = [
    ["Ensembl_geneid",          str],
    ["Ensembl_transcriptid",    str],
    ["Ensembl_proteinid",       str],
    ["Uniprot_acc",             str],
    ["HGVSc_ANNOVAR",           str],
    ["HGVSp_ANNOVAR",           str],
    ["HGVSc_snpEff",            str],
    ["HGVSp_snpEff",            str],
    ["GENCODE_basic",           str],
    ["SIFT_score",              float],
    ["SIFT_pred",               str],
    ["SIFT4G_score",            float],
    ["SIFT4G_pred",             str],
    ["Polyphen2_HDIV_score",    float],
    ["Polyphen2_HDIV_pred",     str],
    ["Polyphen2_HVAR_score",    float],
    ["Polyphen2_HVAR_pred",     str],
    ["MutationAssessor_score",  float],
    ["MutationAssessor_pred",   str],
    ["FATHMM_score",            float],
    ["FATHMM_pred",             str],
    ["MPC_score",               float]
]

ALL_TABS = [VARIANT_TAB, FACET_TAB, TRANSCRIPT_TAB]

#========================================
FLD_NAME_MAP = {
    "ref": "REF",
    "alt": "ALT",
    "Eigen_pred_coding": "Eigen_phred_coding"
}

def _normFieldName(name):
    global FLD_NAME_MAP
    name = name.replace('-', '_')
    return FLD_NAME_MAP.get(name, name)

#========================================
def setupFields(field_line):
    global ALL_TABS, FLD_NAME_MAP
    assert field_line.startswith('#')
    field_names = field_line[1:].split()
    assert field_names[0].startswith("chr")
    assert field_names[1].startswith("pos")
    fields_idxs = {_normFieldName(name): idx
        for idx, name in enumerate(field_names)}
    errors = 0
    for tab in ALL_TABS:
        for field_info in tab:
            idx = fields_idxs.get(field_info[0])
            if idx is None:
                errors += 1
                logging.error("No field registered: %s" % field_info[0])
            else:
                if len(field_info) == 2:
                    field_info.append(idx)
                else:
                    field_info[2] = idx
    if errors > 0:
        logging.info("Available fields:\n=====\n"
            + "\n".join(sorted(fields_idxs.keys())))
    assert errors == 0

#========================================
def iterFields(fields, properties_tab):
    for name, tp, idx in properties_tab:
        val = fields[idx]
        if val == '.':
            yield name, None
        else:
            yield name, tp(val)

def iterDeepFields(fields, properties_tab):
    for name, tp, idx in properties_tab:
        val_seq = []
        for val in fields[idx].split(';'):
            if val == '.':
                val_seq.append(None)
            else:
                val_seq.append(tp(val))
        yield name, val_seq

#========================================
class DataCollector:
    def __init__(self):
        self.mCounts = [0, 0, 0]
        self.mCurRecord = None

    def getCounts(self):
        return self.mCounts

    def ingestLine(self, line):
        global VARIANT_TAB, FACET_TAB, TRANSCRIPT_TAB
        if line.endswith('\n'):
            line = line[:-1]
        fields = line.split('\t')
        chrom = "chr" + str(fields[0])
        pos = int(fields[1])
        new_record = False
        if self.mCurRecord is None or (chrom, pos) != self.mCurRecord[0]:
            new_record = True
        new_variant = new_record

        var_data = dict()
        for name, val in iterFields(fields, VARIANT_TAB):
            var_data[name] = val
            if not new_variant and val != self.mCurRecord[1][-1][name]:
                new_variant = True

        facet_data = {name: val
            for name, val in iterFields(fields, FACET_TAB)}

        tr_data_seq = None
        for name, val_seq in iterDeepFields(fields, TRANSCRIPT_TAB):
            if tr_data_seq is None:
                tr_data_seq = [{name: val} for val in val_seq]
            else:
                for idx, val in enumerate(val_seq):
                    tr_data_seq[idx][name] = val
        if tr_data_seq is None:
            tr_data_seq = []
        facet_data["transcripts"] = tr_data_seq
        self.mCounts[2] += len(tr_data_seq)
        self.mCounts[1] += 1

        ret = None
        if new_record:
            self.mCounts[0] += 1
            var_data["facets"] = [facet_data]
            ret, self.mCurRecord = self.mCurRecord, [(chrom, pos), [var_data]]
        elif new_variant:
            self.mCounts[0] += 1
            var_data["facets"] = [facet_data]
            self.mCurRecord[1].append(var_data)
        else:
            self.mCurRecord[1][-1]["facets"].append(facet_data)

        return ret

    def finishUp(self):
        return self.mCurRecord

#========================================
#========================================
class ReaderDBNSFP4:
    def __init__(self, file_list, chrom_loc = "chr"):
        self.mFiles = extendFileList(file_list)
        self.mChromLoc = chrom_loc

    def read(self):
        exceptions = 0
        for chrom_file in self.mFiles:
            chrom = detectFileChrom(chrom_file, self.mChromLoc)
            logging.info("Evaluation of %s in %s" % (chrom, chrom_file))
            with gzip.open(chrom_file, 'rt') as text_inp:
                time_rep = TimeReport("chr" + chrom)
                collector = DataCollector()
                for line_no, line in enumerate(text_inp):
                    if line_no == 0:
                        setupFields(line)
                        continue
                    try:
                        info = collector.ingestLine(line)
                        if info is not None:
                            yield info
                        if (line_no % 10000) == 0:
                            total_var, _, _ = collector.getCounts()
                            time_rep.portion(total_var)
                    except IndexError:
                        exceptions += 1
                info = collector.finishUp()
                if info:
                    yield info
                total_var, total_facets, total_tr = collector.getCounts()
                time_rep.done(total_var)
                logging.info("transcripts: %d, facets: %d, exceptions: %d"
                    % (total_tr, total_facets, exceptions))

#========================================
def reader_dbNSFP4(properties, schema_h = None):
    return ReaderDBNSFP4(
        properties["file_list"],
        properties.get("chrom_loc", "chr"))


#========================================
if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    reader = reader_dbNSFP4({"file_list": sys.argv[1]})
    dumpReader(reader)
