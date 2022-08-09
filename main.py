import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer 
from pyspark.sql.functions import * 
from pyspark.ml import Pipeline 
from indexers import pipeline
from splitDataset import split_data

spark = SparkSession.builder.master("local[*]").getOrCreate() 

df = spark.read.csv('FIC.Full CSV.csv', header = True, inferSchema = True)
df.printSchema()

df = df.withColumnRenamed("Life.Style                                                                              ","Lifestyle") \
    .withColumnRenamed("Family.History","FamilyHistory") \
    .withColumnRenamed("SK.React","SKReact")

transformed_df = pipeline.fit(df).transform(df) 

cols = ("Gender","Locality","Marital status                       ","Lifestyle","Sleep","Category",
       "Depression","Hyperlipi","Smoking","FamilyHistory","HTN","Allergies",
       "Others ","CO","Diagnosis","Hypersensitivity","SKReact","Mortality")

transformed_df.drop(*cols) \
   .printSchema()

train_set, test_set = split_data(transformed_df)
print("Training Dataset Count: " + str(train_set.count()))
print("Test Dataset Count: " + str(test_set.count()))