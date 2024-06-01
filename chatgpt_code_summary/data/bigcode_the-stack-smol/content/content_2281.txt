# import libraries
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *

from pyspark.sql.functions import col, count, lit, rand, when

import pandas as pd
from math import ceil

#################################################
# spark config
#################################################
mtaMaster = "spark://192.168.0.182:7077"

conf = SparkConf()
conf.setMaster(mtaMaster)

conf.set("spark.executor.memory", "24g")
conf.set("spark.driver.memory", "26g")
conf.set("spark.cores.max", 96)
conf.set("spark.driver.cores", 8)

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer", "256m")
conf.set("spark.kryoserializer.buffer.max", "256m")

conf.set("spark.default.parallelism", 24)

conf.set("spark.eventLog.enabled", "true")
conf.set("spark.eventLog.dir", "hdfs://192.168.0.182:9000/eventlog")
conf.set("spark.history.fs.logDirectory", "hdfs://192.168.0.182:9000/eventlog")

conf.set("spark.driver.maxResultSize", "4g")

conf.getAll()

#################################################
# create spark session
#################################################
spark = SparkSession.builder.appName('ML2_HV_v1_NYT_sim1_and_sim3_to_sim2_round5_human_validation').config(conf=conf).getOrCreate()

sc = spark.sparkContext

# check things are working
print(sc)
print(sc.defaultParallelism)
print("SPARK CONTEXT IS RUNNING")

#################################################
# define major topic codes
#################################################

# major topic codes for loop (NO 23 IN THE NYT CORPUS)
majortopic_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 100]
#majortopic_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 100]

#################################################
# read result data from round 3
#################################################

df_results = spark.read.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v1_NYT_r5_classified.parquet").repartition(50)

# verdict to integer for the comparison with majortopic later
df_results = df_results.withColumn('verdict', df_results.verdict.cast(IntegerType()))

#################################################
# create table to store sample and validation numbers
#################################################

columns = ["num_classified", "num_sample", "num_non_sample", "num_correct", "num_incorrect", "precision_in_sample", "num_added_to_training"]
df_numbers = pd.DataFrame(index=majortopic_codes, columns=columns)
df_numbers = df_numbers.fillna(0)

#################################################
# create table of samples from results
#################################################

# constants for sample size calculation for 95% confidence with +-0.05 precision confidence interval:
z = 1.96
delta = 0.05
z_delta = z*z*0.5*0.5/(delta*delta)
print("z_delta :", z_delta)

for i in majortopic_codes:
    df_classified = df_results.where(col('verdict') == i)
    num_classified = df_classified.count()
    df_numbers["num_classified"].loc[i] = num_classified
    print("MTC:", i, "num_classified: ", num_classified)
    if num_classified > 100:
        sample_size = ceil(z_delta/(1+1/num_classified*(z_delta-1)))
        print("sample_size: ", sample_size)
        if sample_size < 100:
            sample_size = 100
        df_sample = df_classified.sort('doc_id').withColumn('random', rand()).sort('random').limit(sample_size).drop('random')
        df_sample_num = df_sample.count()
        print("df_sample: ", df_sample_num)
        # separate non-sample from sample elements
        ids_drop = df_sample.select("doc_id")
        df_non_sample = df_classified.join(ids_drop, "doc_id", "left_anti")
        df_numbers["num_sample"].loc[i] = df_sample_num
        df_numbers["num_non_sample"].loc[i] = df_non_sample.count()
    else:
        df_numbers["num_sample"].loc[i] = num_classified
        df_sample = df_classified
        df_non_sample = None

    # create table of all samples and add new sample to it
    if i == 1:
        df_sample_all = df_sample
    else:
        df_sample_all = df_sample_all.union(df_sample)
    #print("MTC:", i, "df_sample_all: ", df_sample_all.count())

    # create table of all non-samples and add new non-sample to it
    if i == 1:
        df_non_sample_all = None

    if df_non_sample != None and df_non_sample_all == None:
        df_non_sample_all = df_non_sample
    elif df_non_sample != None and df_non_sample_all != None:
        df_non_sample_all = df_non_sample_all.union(df_non_sample)
    #print("MTC:", i, "df_non_sample_all: ", df_non_sample_all.count())
    print("MTC:", i)


#################################################
# check precision by majortopic codes
#################################################

# count correctly classified and precision for each majortopic code and write to table of numbers
df_correctly_classified = df_sample_all.where(col('majortopic') == col('verdict'))
for i in majortopic_codes:
    num_correct = df_correctly_classified.where(col('verdict') == i).count()
    df_numbers["num_correct"].loc[i] = num_correct
    df_numbers["precision_in_sample"].loc[i] = num_correct/df_numbers["num_sample"].loc[i]

# count incorrectly classified for debugging and checking
df_incorrectly_classified = df_sample_all.where(col('majortopic') != col('verdict'))
for i in majortopic_codes:
    num_incorrect = df_incorrectly_classified.where(col('verdict') == i).count()
    df_numbers["num_incorrect"].loc[i] = num_incorrect

print(df_numbers)


#################################################
# create tables of elements based on precision
#################################################

# create tables for sorting elements based on precision results
# where precision is equal to or greater than 75%
# NOTE: validated wrongly classified elements will NOT be added to the results with the wrong major
# topic code, instead they will be added to the unclassified elements as in rounds 1&2
df_replace_all = None
# where precision is less than 75%
df_non_sample_replace = None
df_correct_replace = None
df_wrong_replace = None

for i in majortopic_codes:
    print("create tables MTC:", i)
    if df_numbers["precision_in_sample"].loc[i] >= 0.75:
        # in this case add all elements from sample and non-sample to the training set with
        # new major topic code i, EXCEPT for validated negatives, those are added to back into the
        # test set
        # first add wrong sample elements to their table
        df_lemma = df_sample_all.where(col('verdict') == i).where(col('majortopic') != col('verdict'))
        if df_wrong_replace == None:
            df_wrong_replace = df_lemma
        else:
            df_wrong_replace = df_wrong_replace.union(df_lemma)
        # get doc_ids for these elements to remove them from the rest of the elements classified as
        # belonging to major topic i
        ids_drop = df_lemma.select("doc_id")
        # get all elements classified as belonging to major topic code i
        df_lemma = df_results.where(col('verdict') == i)
        # remove wrongly classified from df_lemma
        df_lemma = df_lemma.join(ids_drop, "doc_id", "left_anti")
        # add df_lemma to df_replace_all
        if df_replace_all == None:
            df_replace_all = df_lemma
        else:
            df_replace_all = df_replace_all.union(df_lemma)
        # write numbers to df_numbers
        df_numbers["num_added_to_training"].loc[i] = df_lemma.count()
        #print("MTC:", i, "df_replace_all: ", df_replace_all.count())
    else:
        # in this case add only correct elements from sample to training set, the rest go back in
        # the test set
        # first add non-sample elements to their table, BUT we have to check whether non-sample elements
        # exist
        if df_non_sample_all != None:
            df_lemma = df_non_sample_all.where(col('verdict') == i)
            if df_non_sample_replace == None:
                df_non_sample_replace = df_lemma
            else:
                df_non_sample_replace = df_non_sample_replace.union(df_lemma)
        else:
            df_non_sample_replace = None
        #print("MTC:", i, "df_non_sample_replace: ", df_non_sample_replace.count())
        # second add correct sample elements to their table
        df_lemma = df_sample_all.where(col('verdict') == i).where(col('majortopic') == col('verdict'))
        if df_correct_replace == None:
            df_correct_replace = df_lemma
        else:
            df_correct_replace = df_correct_replace.union(df_lemma)
        df_numbers["num_added_to_training"].loc[i] = df_lemma.count()
        #print("MTC:", i, "df_correct_replace: ", df_correct_replace.count())
        # finally add wrong sample elements to their table
        df_lemma = df_sample_all.where(col('verdict') == i).where(col('majortopic') != col('verdict'))
        if df_wrong_replace == None:
            df_wrong_replace = df_lemma
        else:
            df_wrong_replace = df_wrong_replace.union(df_lemma)
        #print("MTC:", i, "df_wrong_replace: ", df_wrong_replace.count())

# sometimes there will be no major topic code with precision => 75%
if df_replace_all == None:
    df_replace_all = "empty"

# sometimes there will be no non-sample elements
if df_non_sample_replace == None:
    df_non_sample_replace = "empty"

# the reason for creating these "empty" values, is because they will persist after we clear the
# cache, and we can use them later in the workflow control

# write all tables to parquet before clearing memory
df_correct_replace.write.parquet("hdfs://192.168.0.182:9000/input/df_correct_replace_temp.parquet", mode="overwrite")
df_wrong_replace.write.parquet("hdfs://192.168.0.182:9000/input/df_wrong_replace_temp.parquet", mode="overwrite")
# sometimes there will be no non-sample elements
if df_non_sample_replace != "empty":
    df_non_sample_replace.write.parquet("hdfs://192.168.0.182:9000/input/df_non_sample_replace_temp.parquet", mode="overwrite")
# sometimes there will be no major topic code with precision => 75%
if df_replace_all != "empty":
    df_replace_all.write.parquet("hdfs://192.168.0.182:9000/input/df_replace_all_temp.parquet", mode="overwrite")

# write df_numbers to csv
df_numbers.to_csv("ML2_HV_v1_NYT_human_validation_numbers_r5.csv", index=True)

# empty memory
spark.catalog.clearCache()
print("cache cleared")

#################################################
# prepare df_original to add tables to it
#################################################

df_original = spark.read.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v1_NYT_r5_train_and_remaining_NOTclassified.parquet").repartition(50)
# we need to create a new majortopic column, because we are now adding back in elements with
# potentially new labels
df_original = df_original.withColumnRenamed('majortopic', 'mtc_after_r4')
df_original = df_original.withColumn('majortopic', df_original['mtc_after_r4'])
# finally, create the new train id column
df_original = df_original.withColumn("train_r6", when(df_original["train_r5"] == 1, 1).otherwise(0))

#################################################
# add df_replace_all back to df_original
#################################################

if df_replace_all != "empty":
    print("df_replace_all is NOT empty")

    df_replace_all = spark.read.parquet("hdfs://192.168.0.182:9000/input/df_replace_all_temp.parquet").repartition(50)
    # we need to create a new majortopic column, because we are now adding back in elements with
    # potentially new labels
    df_replace_all = df_replace_all.withColumnRenamed('majortopic', 'mtc_after_r4')
    df_replace_all = df_replace_all.withColumn('majortopic', df_replace_all['verdict'])
    # create the new train id column
    df_replace_all = df_replace_all.withColumn("train_r6", lit(1))
    # drop the extra columns to be able to add it back to df_original
    df_replace_all = df_replace_all.drop('verdict')

    # add df_replace_all elements to df_original
    df_original = df_original.union(df_replace_all)

else:
    print("df_replace_all is empty")

#################################################
# add df_non_sample_replace back to df_original
#################################################

if df_non_sample_replace != "empty":
    print("df_non_sample_replace is NOT empty")

    df_non_sample_replace = spark.read.parquet("hdfs://192.168.0.182:9000/input/df_non_sample_replace_temp.parquet").repartition(50)
    # we need to create a new majortopic column, because we are now adding back in elements with
    # potentially new labels
    df_non_sample_replace = df_non_sample_replace.withColumnRenamed('majortopic', 'mtc_after_r4')
    df_non_sample_replace = df_non_sample_replace.withColumn('majortopic', df_non_sample_replace['mtc_after_r4'])
    # create the new train id column
    df_non_sample_replace = df_non_sample_replace.withColumn("train_r6", lit(0))
    # drop the extra columns to be able to add it back to df_original
    df_non_sample_replace = df_non_sample_replace.drop('verdict')

    # add df_non_sample_replace elements to df_original
    df_original = df_original.union(df_non_sample_replace)

else:
    print("df_non_sample_replace is empty")

#################################################
# add df_correct_replace back to df_original
#################################################

df_correct_replace = spark.read.parquet("hdfs://192.168.0.182:9000/input/df_correct_replace_temp.parquet").repartition(50)
# we need to create a new majortopic column, because we are now adding back in elements with
# potentially new labels
df_correct_replace = df_correct_replace.withColumnRenamed('majortopic', 'mtc_after_r4')
df_correct_replace = df_correct_replace.withColumn('majortopic', df_correct_replace['verdict'])
# create the new train id column
df_correct_replace = df_correct_replace.withColumn("train_r6", lit(1))
# drop the extra columns to be able to add it back to df_original
df_correct_replace = df_correct_replace.drop('verdict')

# add df_correct_replace elements to df_original
df_original = df_original.union(df_correct_replace)

#################################################
# add df_wrong_replace back to df_original
#################################################

df_wrong_replace = spark.read.parquet("hdfs://192.168.0.182:9000/input/df_wrong_replace_temp.parquet").repartition(50)
# we need to create a new majortopic column, because we are now adding back in elements with
# potentially new labels
df_wrong_replace = df_wrong_replace.withColumnRenamed('majortopic', 'mtc_after_r4')
df_wrong_replace = df_wrong_replace.withColumn('majortopic', df_wrong_replace['mtc_after_r4'])
# create the new train id column
df_wrong_replace = df_wrong_replace.withColumn("train_r6", lit(0))
# drop the extra columns to be able to add it back to df_original
df_wrong_replace = df_wrong_replace.drop('verdict')

# add df_wrong_replace elements to df_original
df_original = df_original.union(df_wrong_replace)

#################################################
# final write operations
#################################################

df_original.write.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v1_NYT_round6_start.parquet", mode="overwrite")

df_original.groupBy("train_r6").count().show(n=30)

# empty memory
spark.catalog.clearCache()
print("cache cleared")

# write to pandas and export to csv for debugging
df_original = spark.read.parquet("hdfs://192.168.0.182:9000/input/ML2_HV_v1_NYT_round6_start.parquet").repartition(50)
df_original = df_original.drop('text', 'words', 'features', 'raw_features').toPandas()
df_original.to_csv("ML2_HV_v1_NYT_round6_starting_table.csv", index=False)

sc.stop()
spark.stop()
