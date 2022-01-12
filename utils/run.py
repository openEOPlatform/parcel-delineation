from pyspark.sql import SparkSession
import cloudpickle
import pyspark.serializers
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.models import Model


def fix_pickle():

    def unpack(model, training_config, weights):
        restored_model = deserialize(model)
        if training_config is not None:
            restored_model.compile(
                **saving_utils.compile_args_from_training_config(
                    training_config
                )
            )
        restored_model.set_weights(weights)
        return restored_model

    # Hotfix function
    def make_keras_picklable():

        def __reduce__(self):
            model_metadata = saving_utils.model_metadata(self)
            training_config = model_metadata.get("training_config", None)
            model = serialize(self)
            weights = self.get_weights()
            return (unpack, (model, training_config, weights))

        cls = Model
        cls.__reduce__ = __reduce__

    # Run the function
    make_keras_picklable()


def get_spark_sql(local=False):
    # initialise sparkContext
    sc = get_spark_context(local=local)
    from pyspark.sql import SQLContext
    sqlContext = SQLContext(sc)
    return sqlContext


def get_spark_context(name="PARCEL", local=False):

    pyspark.serializers.cloudpickle = cloudpickle

    if not local:

        # Hotfix to make keras model pickable
        fix_pickle()

        spark = SparkSession.builder \
            .appName(name) \
            .config('spark.executor.memory', '4G') \
            .config('spark.driver.memory', '8G') \
            .getOrCreate()
        sc = spark.sparkContext
    else:
        spark = SparkSession.builder \
            .appName(name) \
            .master('local[1]') \
            .config('spark.driver.host', '127.0.0.1') \
            .config('spark.executor.memory', '4G') \
            .config('spark.driver.memory', '4G') \
            .getOrCreate()
        sc = spark.sparkContext
    return sc
