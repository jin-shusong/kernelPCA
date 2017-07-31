import scala.math.tanh
class sigmoidKernel(val scale: Double = 1.0, val offset: Double = 1.0) extends MercerKernel {
  val kernelName = "Sigmoid"
  def k(x: org.apache.spark.ml.linalg.DenseVector,
        y: org.apache.spark.ml.linalg.DenseVector): Double = {
    assert(x.size == y.size)
    val tmpMatrix = new org.apache.spark.ml.linalg.DenseMatrix(x.size, 1, x.toArray)
    val tmpRes = tmpMatrix.transpose.multiply(y)
    tanh(scale * tmpRes(0) + offset)
  }
}
