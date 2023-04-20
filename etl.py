import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """ Staging, Extracting song data json files from AWS S3 and Writing Extracted parquet data file 
        back to AWS S3
    
        Arguments:
            spark {object}:         The entry point to programming Spark with the Dataset and
                                    DataFrame API.
            input_data {string}:    S3 bucket where Sparkify's event data is stored 
            output_data {string}:   S3 bucket to store extracted parquet data file
    
        Returns:
            No return values    
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data+'songs/'+'songs.parquet', 
                              partitionBy==['year','artist_id'])

    # extract columns to create artists table
    artists_table = df.select(["artist_id", "artist_name", "artist_location", 
                               "artist_latitude", "artist_longitude"]).distinct()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists/' + 'artists.parquet',
                                partitionBy=['artist_id'] )


def process_log_data(spark, input_data, output_data):
    """ Staging, Extracting log data json files from AWS S3 and Writing Extracted parquet 
        data file back to AWS S3.
    
        Arguments:
            spark {object}:         The entry point to programming Spark with the Dataset 
                                    and
                                    DataFrame API.
            input_data {string}:    S3 bucket path where Sparkify's event data is stored 
            output_data {string}:   S3 bucket path used to store extracted parquet data 
                                    file
        
        Returns:
            No return values
    """
    
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where('page="NextSong"')

    # extract columns for users table    
    users_table = df.select(["userId", "firstName", "lastName", "gender",
                             "level"]).distinct()
    
    # write users table to parquet files
    users_table.write.parquet(output_data + 'users/' + 'users.parquet', partitionBy =
                              ['userId'])

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    df = df.withColumn('datetime', get_datetime(df.ts))
    
    # extract columns to create time table
    time_table = df.select('datetime') \
                           .withColumn('start_time', df.datetime) \
                           .withColumn('hour', hour('datetime')) \
                           .withColumn('day', dayofmonth('datetime')) \
                           .withColumn('week', weekofyear('datetime')) \
                           .withColumn('month', month('datetime')) \
                           .withColumn('year', year('datetime')) \
                           .withColumn('weekday', dayofweek('datetime')) \
                           .dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + 'time/' + 'time.parquet', partitionBy=
                             ['start_time'])

    # read in song data to use for songplays table
    song_df = spark.read.json(input_data+'song_data/*/*/*/*.json')

    # extract columns from joined song and log datasets to create songplays table 
    merged_df = df.join(song_df, col('df.artist') == col('song_df.artist_name'), 'inner')
    songplays_table = merged_df.select(
        col('df.datetime').alias('start_time'),
        col('df.userId').alias('user_id'),
        col('df.level').alias('level'),
        col('song_df.song_id').alias('song_id'),
        col('song_df.artist_id').alias('artist_id'),
        col('df.sessionId').alias('session_id'),
        col('df.location').alias('location'), 
        col('df.userAgent').alias('user_agent'),
        year('df.datetime').alias('year'),
        month('df.datetime').alias('month')) \
        .withColumn('songplay_id', monotonically_increasing_id())
    
    # write songplays table to parquet files partitioned by year and month
     songplays_table.write.parquet(output_data + 'songplays/' + 'songplays.parquet',partitionBy=['start_time', 'user_id'])


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-data-lake/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
