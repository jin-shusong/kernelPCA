/**
  * Created by jinss on 7/5/17.
  */

import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix}
import scala.math.tanh

class tanhKernel(val scale: Double = 1.0, val offset: Double = 1.0) extends MercerKernel {
  val kernelName = "Tanh"
  def k(x: org.apache.spark.ml.linalg.DenseVector,
        y: org.apache.spark.ml.linalg.DenseVector): Double = {
    assert(x.size == y.size)
    val tmpMatrix = new org.apache.spark.ml.linalg.DenseMatrix(x.size, 1, x.toArray)
    val tmpRes = tmpMatrix.transpose.multiply(y)
    tanh(scale * tmpRes(0) + offset)
  }
}
