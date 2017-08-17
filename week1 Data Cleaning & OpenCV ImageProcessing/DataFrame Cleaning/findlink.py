# Calculate the average rating of each genre
# Note: this is a python-spark implementation of the java-spark code shown in the lecture (MovieLensLarge)
# In order to run this, we use spark-submit, but with a different argument
# pyspark  \
#   --master yarn-client \
#   --num-executors 3 \
#   average_rating_per_genre.py

from pyspark import SparkContext
from random import uniform
from math import sqrt


# This function convert rawdata into the dataset including three float coordinates
# 17 data -> ly6c, cd11b, sca1
def extract_dataset(record):
  try:
    if(len(record.split(",")) == 6):
       order, id, part, label, link, nn = record.split(",")
       if (label == "Image") and (link != ""):
        lowerpart= str(part).lower()
        return [lowerpart+","+link]
    else:
           return []
  except:
    return []


def toCSVLine(data):
    return '\t'.join(str(d) for d in data)

if __name__ == "__main__":
  sc = SparkContext(appName="k-means clustering")
  raw_data = sc.textFile("C://MainAppDisk/USYD/wipehero/wipeheroraw.csv")
  # k=5
  # iter_max=10
  data_list=raw_data.map(extract_dataset).filter(lambda rec : rec!=[])
  new_data=data_list.filter(lambda rec : rec != None)

  sorted= new_data.sortBy(lambda rec: rec[0], ascending=True, numPartitions=1)
  csvoutput=sorted.map(toCSVLine)
  csvoutput.coalesce(1, True).saveAsTextFile("C:/MainAppDisk/USYD/wipehero/dddd")