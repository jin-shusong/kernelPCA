/**
  * Created by jinss on 7/5/17.
  */

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.{DenseMatrix, DenseVector, eigSym, tile}
import breeze.numerics._

class kernelPCA(val data: org.apache.spark.ml.linalg.DenseMatrix, //输入的数据，以矩阵形式提供
                val kernelName: String, val para: Array[Double], //核的名字
                val desireFeatures: Int = 0, //准备保留的特征数，必须小于矩阵的列数，特征数为0表明由threshold选择数量
                val threshold: Double = 1e-4 // 阈值，特征值小于的特征向量将被舍弃
               ) {
  require(desireFeatures.isInstanceOf[Int])
  require(desireFeatures >= 0)
  private val tmpname = kernelName.toLowerCase
  val kernelFunction = tmpname match {
    case "gaussian" => {
      require(para.length == 1)
      new GaussianKernel(para(0))
    }
    case "sigmoid" => {
      assert(para.length == 2)
      new sigmoidKernel(para(0), para(1))
    }
    case "tanh" => {
      assert(para.length == 2)
      new tanhKernel(para(0), para(1))
    }
    case "polynomial" => {
      assert(para.length == 3)
      new polynomialKernel(para(0), para(1), para(2))
    }
    case _ => {
      println("Use Gaussian Kernel with Sigma 1")
      new GaussianKernel(1.0)
    }
  }
  val km = myUtilForKPCA.kernelMatrix(kernelFunction,data, center = true)
  private val tmpMatrix1 = myUtilForKPCA.matrixScale(km, 1.0 / data.numRows)
  private val tmpMatrix2 = new breeze.linalg.DenseMatrix(tmpMatrix1.numRows, tmpMatrix1.numCols, tmpMatrix1.toArray)
  private val tmpEigSym = eigSym(tmpMatrix2)
  private val tmpEigValue = tmpEigSym.eigenvalues((data.numRows - 1) to 0 by -1)
  private val tmpEigMatrix = myUtilForKPCA.reverseMatrixByColumn(tmpEigSym.eigenvectors)
  val numOfFeatures = if (desireFeatures == 0) tmpEigValue.toArray.filter(_ > threshold).length else desireFeatures
  private val breezeEigVectors = tmpEigMatrix(::, 0 until numOfFeatures)
  private val breezeEigValue = tmpEigValue(0 until numOfFeatures)
  val EigenValues = myUtilForKPCA.breezeDenseVectorToSparkDenseVector(breezeEigValue)
  val EigenVectors = myUtilForKPCA.breezeDenseMatrixToSparkDenseMatrix(breezeEigVectors)
  val principalComponentVectors = findPCVectors(breezeEigVectors, breezeEigValue) //包含主成分向量的矩阵
  val rotated = km.multiply(principalComponentVectors) //原数据在主成分上的投影
  val xMatrix = data

  private def findPCVectors(eigVector: breeze.linalg.DenseMatrix[Double],
                            eigValue: breeze.linalg.DenseVector[Double]
                           ): org.apache.spark.ml.linalg.DenseMatrix = {
    val numRows = eigVector.rows
    val numCols = eigVector.cols
    assert(numCols == eigValue.length)
    val tmpMatrix = new breeze.linalg.DenseMatrix(1, numCols, sqrt(eigValue).toArray)
    val tmpMatrix2 = tile(tmpMatrix, numRows, 1)
    val tmpMatrix3 = eigVector /:/ tmpMatrix2
    myUtilForKPCA.breezeDenseMatrixToSparkDenseMatrix(tmpMatrix3)
  }

  def predict(x: org.apache.spark.ml.linalg.DenseMatrix): org.apache.spark.ml.linalg.DenseMatrix = {
    val n = x.numRows
    val m = data.numRows
    val knc = myUtilForKPCA.kernelMatrix(kernelFunction,x, xMatrix)
    val ka = myUtilForKPCA.kernelMatrix(kernelFunction,xMatrix, center = false)
    val tmp = myUtilForKPCA.centralizeKernelMatrix(knc, ka)
    tmp.multiply(principalComponentVectors)
  }
}