import org.apache.spark.ml.linalg.{DenseVector, DenseMatrix}
import scala.math._

class GaussianKernel(val sigma: Double) extends MercerKernel {
  if (sigma <= 0) throw new IllegalArgumentException("sigma is not positive.")
  val kernelName = "Gaussian"

  def k(x: org.apache.spark.ml.linalg.DenseVector,
        y: org.apache.spark.ml.linalg.DenseVector): Double = {
    assert(x.size == y.size)
    val xMatrix = new org.apache.spark.ml.linalg.DenseMatrix(x.size, 1, x.toArray)
    val yMatrix = new org.apache.spark.ml.linalg.DenseMatrix(y.size, 1, y.toArray)
    val tmp1 = (xMatrix.transpose.multiply(y)) (0)
    val tmp2 = (xMatrix.transpose.multiply(x)) (0)
    val tmp3 = (yMatrix.transpose.multiply(y)) (0)
    math.exp(sigma * (2.0 * tmp1 - tmp2 - tmp3))
  }
}
