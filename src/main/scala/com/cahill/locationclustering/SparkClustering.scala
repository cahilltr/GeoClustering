package com.cahill.locationclustering

import breeze.linalg.{DenseMatrix, DenseVector}
import nak.cluster.Kmeans
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SparkSession}
import nak.cluster._
import org.apache.spark.rdd.RDD


// From: https://www.oreilly.com/ideas/clustering-geolocated-data-using-spark-and-dbscan
object SparkClustering {

	def main(args:Array[String]): Unit = {
		val spark = SparkSession
			.builder()
			.master("local[*]")
			.appName("Spark Location Clustering")
			.getOrCreate()
		val sc: SparkContext = spark.sparkContext

		val singleUserInput = spark.read
			.format("org.apache.spark.csv")
			.option("header", true)
			.option("inferSchema", true)
  		.csv(Thread.currentThread().getContextClassLoader.getResource("1ab974d2-72cc-481e-b07c-8a41bdad2963.csv").getPath)

		val pairRDD:RDD[(String, DenseMatrix[Double])] = singleUserInput
			.select("advertiser_id", "latitude", "longitude")
			.rdd.groupBy(r => r(0).toString).map(kv => {
			(kv._1, rowsToDenseMatrix(kv._2))
		})

		val clustersRDD = pairRDD.map(kv => (kv._1, nakDBscan(kv._2)))

		val avgClusterPoints = clustersRDD.flatMap(kv => kv._2.map(c => (kv._1, c))).map(kv => (kv._1, averageCluster(kv._2)))

		avgClusterPoints.foreach(println)

		sc.stop()
		spark.stop()
	}

	def averageCluster(cluster:GDBSCAN.Cluster[Double]):(Double, Double) = {
		val xList = cluster.points.map(p => p.value.data(0).doubleValue()).toList
		val xAvg = xList.sum/xList.length

		val yList = cluster.points.map(p => p.value.data(1).doubleValue()).toList
		val yAvg = yList.sum/yList.length
		(xAvg, yAvg)
	}

	// https://www.programcreek.com/scala/breeze.linalg.DenseMatrix
	def rowsToDenseMatrix(rows:Iterable[Row]):DenseMatrix[Double] = {
		val dvArray:Array[DenseVector[Double]] = rows.map(row => DenseVector(rowToArray(row))).toArray
		val arrayValues = dvArray.flatMap(dv => dv.data)
		new DenseMatrix[Double](dvArray.length, dvArray(0).length, arrayValues)
	}

	def rowToArray(row:Row):Array[Double] = {
		Array[Double](row.getAs[Double](1), row.getAs[Double](2))
	}

	def nakDBscan(v : breeze.linalg.DenseMatrix[Double]) = {
		val gdbscan = new GDBSCAN(

			DBSCAN.getNeighbours(epsilon = 0.001, distance = Kmeans.euclideanDistance),
			DBSCAN.isCorePoint(minPoints = 3)
		)

		gdbscan.cluster(v)
	}

}
