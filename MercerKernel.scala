/**
  * Created by jinss on 7/5/17.
  */
abstract class MercerKernel {
  val kernelName: String
  def k(x: org.apache.spark.ml.linalg.DenseVector,
        y: org.apache.spark.ml.linalg.DenseVector): Double
}
