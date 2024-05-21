from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col,expr
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import os
import logging
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

# Initialize SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("StreamingConsumerWithCassandra") \
    .config("spark.cassandra.connection.host", "cassandra") \
    .config("spark.cassandra.connection.port", "9042") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1,com.datastax.spark:spark-cassandra-connector_2.12:3.1.0") \
    .getOrCreate()


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')





# Define the checkpoint directory and output directory paths
checkpoint_dir = "/opt/bitnami/spark/checkpoint"
output_dir = "/opt/bitnami/spark/output"

# Create directories if they don't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the schema for the CSV data
schema = StructType([
    StructField("State", IntegerType()),
    StructField("Account_length", IntegerType()),
    StructField("Area_code", IntegerType()),
    StructField("International_plan", IntegerType()),
    StructField("Voice_mail_plan", IntegerType()),
    StructField("Number_vmail_messages", IntegerType()),
    StructField("Total_day_minutes", DoubleType()),
    StructField("Total_day_calls", IntegerType()),
    StructField("Total_day_charge", DoubleType()),
    StructField("Total_eve_minutes", DoubleType()),
    StructField("Total_eve_calls", IntegerType()),
    StructField("Total_eve_charge", DoubleType()),
    StructField("Total_night_minutes", DoubleType()),
    StructField("Total_night_calls", IntegerType()),
    StructField("Total_night_charge", DoubleType()),
    StructField("Total_intl_minutes", DoubleType()),
    StructField("Total_intl_calls", IntegerType()),
    StructField("Total_intl_charge", DoubleType()),
    StructField("Customer_service_calls", IntegerType()),
    StructField("churn", IntegerType())  # Assuming there's a "churn" column in your data
])

# Load the trained model
model_path = "RandomForest"  # Update with the path to your trained model
gbt_model = RandomForestClassificationModel.load(model_path)

# Read data from Kafka as a structured stream
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "kafka_churg") \
    .option("startingOffsets", "latest") \
    .load()

# Parse CSV and apply schema
df = df.selectExpr("CAST(value AS STRING) AS csv_string") \
    .select(split("csv_string", ",").alias("csv_array"))

# Apply schema and cast to appropriate types
for i, field in enumerate(schema):
    df = df.withColumn(field.name, col("csv_array")[i].cast(field.dataType))

# Filter out records with missing or malformed data
# df = df.na.drop()

# Define the feature columns
feature_columns = [
     "International_plan","Voice_mail_plan", "Number_vmail_messages", "Total_day_minutes","Total_day_calls",  "Total_day_charge",
    "Total_eve_minutes",  "Total_eve_charge","Total_eve_calls", "Total_night_minutes","Total_night_calls",
     "Total_night_charge", "Total_intl_minutes","Total_intl_calls","Total_intl_charge",
     "Customer_service_calls"
]

# Initialize VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
def predict_data(df_batch, batch_id):
    try:
        logging.info(f"Processing batch {batch_id}")
        if df_batch.count() > 0:
            df_features = assembler.transform(df_batch)
            predictions = gbt_model.transform(df_features)
            predictions_to_save = predictions.selectExpr(
                "uuid() as id",
                "prediction",
                "cast(State as int) as state",
                "cast(Account_length as int) as account_length",
                "cast(Area_code as int) as area_code",
                "cast(International_plan as int) as international_plan",
                "cast(Voice_mail_plan as int) as voice_mail_plan",
                "cast(Number_vmail_messages as int) as number_vmail_messages",
                "cast(Total_day_minutes as double) as total_day_minutes",
                "cast(Total_day_calls as int) as total_day_calls",
                "cast(Total_day_charge as double) as total_day_charge",
                "cast(Total_eve_minutes as double) as total_eve_minutes",
                "cast(Total_eve_calls as int) as total_eve_calls",
                "cast(Total_eve_charge as double) as total_eve_charge",
                "cast(Total_night_minutes as double) as total_night_minutes",
                "cast(Total_night_calls as int) as total_night_calls",
                "cast(Total_night_charge as double) as total_night_charge",
                "cast(Total_intl_minutes as double) as total_intl_minutes",
                "cast(Total_intl_calls as int) as total_intl_calls",
                "cast(Total_intl_charge as double) as total_intl_charge",
                "cast(Customer_service_calls as int) as customer_service_calls",
                "cast(churn as int) as churn"
            )
            predictions_to_save.write \
                .format("org.apache.spark.sql.cassandra") \
                .mode("append") \
                .option("keyspace", "telecom_data") \
                .option("table", "customer_churn") \
                .save()
            logging.info("Batch processed and data written to Cassandra.")
        else:
            logging.info("Received empty batch.")
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {str(e)}")

# Stream setup
query = df.writeStream \
    .foreachBatch(predict_data) \
    .option("checkpointLocation", checkpoint_dir) \
    .outputMode("append") \
    .start()

query.awaitTermination()
