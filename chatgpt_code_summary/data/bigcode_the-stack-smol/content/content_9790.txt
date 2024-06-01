from itertools import chain
import logging
import sys

from pyspark.sql import functions as f
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)


input_path = '/data/raw/'
output_path = '/data/data_science/powerBI/'


def write_output(df):
    logger.info("CREATING MASTER DATASET")
    logger.info("WRITING: {}".format(output_path + "data_validation_with_diag.parquet"))
    df.write.mode('overwrite').parquet(output_path + 'data_validation_with_diag.parquet')
    return df
		

def main():
  
    pcp_hcc_dropped = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/PCP_HCC_dropped.csv', header=True, sep='|')
    NW_diab = spark.read.parquet("/data/data_science/powerBI/NW_diab_cmd_memb_level.parquet")
    forever = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/ICD10-ForeverCodes.csv', header=True)
    forever = forever.withColumnRenamed('ICD10 Code', 'ICD10_Code')
    forever = forever.withColumnRenamed('Forever Code', 'Forever_Code')
    NW_diab = NW_diab.select('BENE_MBI_ID', f.explode(f.col('diagnosis_list')).alias('diagnosis_code'), 'claim_year')
    diag_forever = NW_diab.join(forever, NW_diab.diagnosis_code == forever.ICD10_Code, how='left')

    diag_forever = diag_forever.select('BENE_MBI_ID', 'diagnosis_code', 'claim_year', 'Forever_Code')
    diag_forever = diag_forever.filter(f.col('claim_year')>='2018')
    diag_forever = diag_forever.filter(~(f.col('diagnosis_code')==''))
    
    icd_hcc = spark.read.csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/ICD10-HCC.csv', header=True)
    icd_hcc = icd_hcc.select('ICD10', 'HCC') 
    diag_for_hcc = diag_forever.join(icd_hcc, diag_forever.diagnosis_code == icd_hcc.ICD10, how='left').drop(icd_hcc.ICD10)   
    diag_for_hcc = diag_for_hcc.filter(~(f.col('HCC').isNull()))
    
    pcp_hcc_dropped = pcp_hcc_dropped.select('BENE_MBI_ID', 'claim_year', 'FINAL_PCP_NPI')
    df_final = pcp_hcc_dropped.join(diag_for_hcc, on=['BENE_MBI_ID', 'claim_year'], how='left')
    df_final = df_final.drop_duplicates()


    write_output(df20)
    df20.coalesce(1).write.mode('overwrite').option("header", "true").csv('wasbs://rdp-uploads@coretechsnmdev.blob.core.windows.net/data_validation_with_diag.csv')	
	
if __name__ == "__main__":
	
    logger.info('START')
    main()
    logger.info('END') 
