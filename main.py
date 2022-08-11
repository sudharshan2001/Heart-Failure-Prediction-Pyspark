import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StringIndexer 
from pyspark.sql.functions import * 
from pyspark.ml import Pipeline 
from script.indexers import pipeline
from script.splitDataset import split_data
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
import os

spark = SparkSession.builder.master("local[*]").getOrCreate() 

df = spark.read.csv('FIC.Full CSV.csv', header = True, inferSchema = True)
df.printSchema()

# Renaming Column with spaces and dot in it
df = df.withColumnRenamed("Life.Style                                                                              ","Lifestyle") \
    .withColumnRenamed("Family.History","FamilyHistory") \
    .withColumnRenamed("SK.React","SKReact")

# Transforming the data after string indexing
transformed_df = pipeline.fit(df).transform(df) 

# Dropping transformed data
cols = ("Gender","Locality","Marital status                       ","Lifestyle","Sleep","Category",
       "Depression","Hyperlipi","Smoking","FamilyHistory","HTN","Allergies",
       "Others ","CO","Diagnosis","Hypersensitivity","SKReact","Mortality")

transformed_df.drop(*cols) \
   .printSchema()

# Chose 3  feautures that impact the label most
features  = ['Age', 'chol', 'thalach']

# Prepping to fit the model
va = VectorAssembler(inputCols = features, outputCol='features')

Vecdf = va.transform(transformed_df)
Vecdf = Vecdf.select(['features', 'label'])

# Splitting the data
train_set, test_set = split_data(Vecdf)
print("Training Dataset Count: " + str(train_set.count()))
print("Test Dataset Count: " + str(test_set.count()))

# Selecting our model and fine tuning it
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')

# Parameters to tune
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [20,30,40]) \
    .addGrid(rf.maxDepth, [5,10,15]) \
    .addGrid(rf.featureSubsetStrategy, [x for x in ["sqrt", "log2", "onethird"]]) \
    .addGrid(rf.impurity, [x for x in ['gini','entropy']]) \
    .addGrid(rf.maxBins, [30, 32, 34, 36]) \
    .build()

# Evaluator
evaluator = BinaryClassificationEvaluator()

# Cross Validation with 3 folds
rf_CV = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator,
                          numFolds=3)

Model = rf_CV.fit(train_set)

predictions = Model.transform(test_set)

print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# Saving our model
os.mkdir('./model1')

Model.write().overwrite().save('./model1')
