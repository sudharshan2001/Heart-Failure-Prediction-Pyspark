from pyspark.ml.feature import StringIndexer 
from pyspark.ml import Pipeline 

indexers = [
StringIndexer(inputCol="Gender", outputCol = "Gender_tr"),  
StringIndexer(inputCol="Locality  ", outputCol = "Locality_tr"),  
StringIndexer(inputCol="Marital status                       ", outputCol = "Marital_tr"),  
StringIndexer(inputCol="Lifestyle", outputCol = "Lifestyle_tr"),
StringIndexer(inputCol="Sleep", outputCol = "Sleep_tr"),  
StringIndexer(inputCol="Category", outputCol = "Category_tr"),  
StringIndexer(inputCol="Depression", outputCol = "Depression_tr"),  
StringIndexer(inputCol="Hyperlipi", outputCol = "Hyperlipi_tr"),
StringIndexer(inputCol="Smoking", outputCol = "Smoking_tr"),  
StringIndexer(inputCol="FamilyHistory", outputCol = "Family.History_tr"),  
StringIndexer(inputCol="HTN", outputCol = "HTN_tr"),  
StringIndexer(inputCol="Allergies", outputCol = "Allergies_tr"),
StringIndexer(inputCol="Others ", outputCol = "Others_tr"),  
StringIndexer(inputCol="CO", outputCol = "CO_tr"),  
StringIndexer(inputCol="Diagnosis", outputCol = "Diagnosis_tr"),  
StringIndexer(inputCol="Hypersensitivity", outputCol = "Hypersensitivity_tr"),
StringIndexer(inputCol="SKReact", outputCol = "SK.React_tr"),
StringIndexer(inputCol="Mortality", outputCol = "label")
]

pipeline = Pipeline(stages=indexers) 
