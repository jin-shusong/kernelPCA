/**
  * Created by jinss on 7/5/17.
  */

import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.distribution.UniformIntegerDistribution

object myUtilForKPCA {
  def vectorSum(vec: org.apache.spark.ml.linalg.DenseVector): Double = {
    val vecArray = vec.toArray
    vecArray.sum
  }

  def colSum(matrix: org.apache.spark.ml.linalg.DenseMatrix): org.apache.spark.ml.linalg.DenseVector = {
    val tmp = for (vec <- matrix.colIter) yield vectorSum(vec.toDense)
    new org.apache.spark.ml.linalg.DenseVector(tmp.toArray)
  }

  def rowSum(matrix: org.apache.spark.ml.linalg.DenseMatrix): org.apache.spark.ml.linalg.DenseVector = {
    val tmp = for (vec <- matrix.rowIter) yield vectorSum(vec.toDense)
    new org.apache.spark.ml.linalg.DenseVector(tmp.toArray)
  }

  def matrixMinusVector(matrix: org.apache.spark.ml.linalg.DenseMatrix,
                        vec: org.apache.spark.ml.linalg.DenseVector): org.apache.spark.ml.linalg.DenseMatrix = {
    val tmp1 = matrix.toArray
    val tmp2 = vec.toArray
    val matrixLength = tmp1.length
    val vecLength = tmp2.length
    var i = 0
    while (i < matrixLength) {
      var j = 0
      while (j < vecLength && i < matrixLength) {
        tmp1(i) -= tmp2(j)
        j += 1
        i += 1
      }
    }
    new org.apache.spark.ml.linalg.DenseMatrix(matrix.numRows, matrix.numCols, tmp1)
  }

  def vecScale(vec: org.apache.spark.ml.linalg.DenseVector,
               x: Double): org.apache.spark.ml.linalg.DenseVector = {
    assert(x != 0.0)
    val tmp = vec.toArray
    val tmp2 = for (y <- tmp) yield y * x
    new org.apache.spark.ml.linalg.DenseVector(tmp2)
  }

  def vecScale(vec: breeze.linalg.DenseVector[Double], x: Double): breeze.linalg.DenseVector[Double] = {
    assert(x != 0.0)
    val tmp = vec.toArray
    val tmp2 = for (y <- tmp) yield y * x
    new breeze.linalg.DenseVector(tmp2)
  }

  def matrixScale(matrix: org.apache.spark.ml.linalg.DenseMatrix,
                  x: Double): org.apache.spark.ml.linalg.DenseMatrix = {
    assert(x != 0)
    val tmp = matrix.toArray
    val tmp2 = for (y <- tmp) yield y * x
    new org.apache.spark.ml.linalg.DenseMatrix(matrix.numRows, matrix.numCols, tmp2)
  }

  def matrixScale(matrix: breeze.linalg.DenseMatrix[Double],
                  x: Double): breeze.linalg.DenseMatrix[Double] = {
    assert(x != 0)
    val tmp = matrix.toArray
    val tmp2 = for (y <- tmp) yield y * x
    new breeze.linalg.DenseMatrix(matrix.rows, matrix.cols, tmp2)
  }

  def centralizeKernelMatrix(matrix: org.apache.spark.ml.linalg.DenseMatrix
                            ): org.apache.spark.ml.linalg.DenseMatrix = {
    val colsum = colSum(matrix)
    val rowsum = rowSum(matrix)
    val nrow = matrix.numRows
    val tmp1 = vecScale(colsum, 1.0 / nrow)
    val tmp2 = vecScale(rowsum, 1.0 / nrow)
    val tmp3 = matrixMinusVector(matrix, tmp1).transpose
    val tmp4 = matrixMinusVector(tmp3, tmp2).transpose
    val tmp5 = vectorSum(colsum)
    val tmp6 = (tmp4.toArray).map(x => x + tmp5 / (nrow * nrow))
    new org.apache.spark.ml.linalg.DenseMatrix(matrix.numRows, matrix.numCols, tmp6)
  }
  def centralizeKernelMatrix(matrix1: org.apache.spark.ml.linalg.DenseMatrix,
                             matrix2: org.apache.spark.ml.linalg.DenseMatrix
                            ): org.apache.spark.ml.linalg.DenseMatrix = {
    val rowsum1 = rowSum(matrix1)
    val nrow1 = matrix1.numRows
    val nrow2 = matrix2.numRows
    val rowsum1scale = vecScale(rowsum1, 1.0/nrow2)
    val tmp1 = matrixMinusVector(matrix1, rowsum1scale)
    val tmp1t= tmp1.transpose
    val rowsum2 = rowSum(matrix2)
    val rowsum2scale = vecScale(rowsum2, 1.0/nrow2)
    val tmp2 = matrixMinusVector(tmp1t, rowsum2scale)
    val tmp2t=tmp2.transpose
    val sumMatrix2= vectorSum(rowsum2)
    val tmp3 = (tmp2t.toArray).map(x => x+sumMatrix2/(nrow1*nrow2) )

    new org.apache.spark.ml.linalg.DenseMatrix(nrow1,nrow2, tmp3)
  }
  def reverseMatrixByColumn(matrix: org.apache.spark.ml.linalg.DenseMatrix
                           ): org.apache.spark.ml.linalg.DenseMatrix = {
    val ncol = matrix.numCols
    val reEye = org.apache.spark.ml.linalg.Vectors.zeros(ncol * ncol).toArray
    for (i <- 0 until ncol) reEye((i + 1) * ncol - i - 1) = 1
    val tmp = new org.apache.spark.ml.linalg.DenseMatrix(ncol, ncol, reEye)
    matrix.multiply(tmp)
  }

  def reverseMatrixByColumn(matrix: breeze.linalg.DenseMatrix[Double]
                           ): breeze.linalg.DenseMatrix[Double] = {
    val ncol = matrix.cols
    matrix(::, (ncol - 1) to 0 by -1)
  }

  def breezeDenseMatrixToSparkDenseMatrix(matrix: breeze.linalg.DenseMatrix[Double]
                                         ): org.apache.spark.ml.linalg.DenseMatrix = {
    val nrow = matrix.rows
    val ncol = matrix.cols
    new org.apache.spark.ml.linalg.DenseMatrix(nrow, ncol, matrix.toArray)
  }

  def sparkDenseMatrixToBreezeDenseMatrix(matrix: org.apache.spark.ml.linalg.DenseMatrix
                                         ): breeze.linalg.DenseMatrix[Double] = {
    val nrow = matrix.numRows
    val ncol = matrix.numCols
    new breeze.linalg.DenseMatrix[Double](nrow, ncol, matrix.toArray)
  }

  def breezeDenseVectorToSparkDenseVector(vec: breeze.linalg.DenseVector[Double]
                                         ): org.apache.spark.ml.linalg.DenseVector = {
    new org.apache.spark.ml.linalg.DenseVector(vec.toArray)
  }

  def sparkDenseVectorToBreezeDenseVector(vec: org.apache.spark.ml.linalg.DenseVector
                                         ): breeze.linalg.DenseVector[Double] = {
    new breeze.linalg.DenseVector[Double](vec.toArray)
  }

  def breezeMatrixSlicing(matrix: breeze.linalg.DenseMatrix[Double],
                          rowIndex: Array[Int] = null,
                          colIndex: Array[Int] = null): breeze.linalg.DenseMatrix[Double] = {
    require(rowIndex != null || colIndex != null)
    val rowLimit = matrix.rows
    val colLimit = matrix.cols
    val tmpRowIndex: Array[Int] = if (rowIndex == null) (0 until rowLimit).toArray else rowIndex
    val tmpColIndex: Array[Int] = if (colIndex == null) (0 until colLimit).toArray else colIndex
    val testRow = tmpRowIndex.filter(math.abs(_) >= rowLimit).length
    val testCol = tmpColIndex.filter(math.abs(_) >= colLimit).length
    require(testRow == 0 && testCol == 0)
    val trueRowIndex = tmpRowIndex.map(x => if (x >= 0) x else (rowLimit + x))
    val trueColIndex = tmpColIndex.map(x => if (x >= 0) x else (colLimit + x))
    val tmpVector = org.apache.spark.ml.linalg.Vectors.zeros(trueRowIndex.length * trueColIndex.length).toArray
    val myTupleArray = for (j <- trueColIndex; i <- trueRowIndex) yield (i, j)
    (0 until tmpVector.length).map(i => tmpVector(i) = matrix(myTupleArray(i)._1, myTupleArray(i)._2))
    new breeze.linalg.DenseMatrix(trueRowIndex.length, trueColIndex.length, tmpVector)
  }

  def sparkMatrixSlicing(matrix: org.apache.spark.ml.linalg.DenseMatrix,
                         rowIndex: Array[Int] = null,
                         colIndex: Array[Int] = null): org.apache.spark.ml.linalg.DenseMatrix = {
    require(rowIndex != null || colIndex != null)
    val rowLimit = matrix.numRows
    val colLimit = matrix.numCols
    val tmpRowIndex: Array[Int] = if (rowIndex == null) (0 until rowLimit).toArray else rowIndex
    val tmpColIndex: Array[Int] = if (colIndex == null) (0 until colLimit).toArray else colIndex
    val testRow = tmpRowIndex.filter(math.abs(_) >= rowLimit).length
    val testCol = tmpColIndex.filter(math.abs(_) >= colLimit).length
    require(testRow == 0 && testCol == 0)
    val trueRowIndex = tmpRowIndex.map(x => if (x >= 0) x else (rowLimit + x))
    val trueColIndex = tmpColIndex.map(x => if (x >= 0) x else (colLimit + x))
    val tmpVector = org.apache.spark.ml.linalg.Vectors.zeros(trueRowIndex.length * trueColIndex.length).toArray
    val myTupleArray = for (j <- trueColIndex; i <- trueRowIndex) yield (i, j)
    (0 until tmpVector.length).map(i => tmpVector(i) = matrix(myTupleArray(i)._1, myTupleArray(i)._2))
    new org.apache.spark.ml.linalg.DenseMatrix(trueRowIndex.length, trueColIndex.length, tmpVector)
  }

  def sampling(lowerLimit: Int,
               upperLimit: Int,
               numOfSamples: Int,
               withoutReplacement: Boolean = true): Array[Int] = {
    require(lowerLimit < upperLimit)

    if (withoutReplacement) require(math.abs(upperLimit - lowerLimit) > 2 * numOfSamples)
    val uf = new UniformIntegerDistribution(lowerLimit, upperLimit)
    val choiceVector: Array[Int] =
      if (withoutReplacement) {
        val t1 = for (i <- 0 until numOfSamples) yield uf.sample()
        var t1List = t1.toList
        while (t1List.toSet.size < numOfSamples) {
          val t2 = for (i <- 0 until numOfSamples) yield uf.sample()
          t1List = t2.toList ::: t1List
        }
        val tmpVector = t1List.toSet.toArray
        val tmpVector2 = for (i <- 0 until numOfSamples) yield tmpVector(i)
        tmpVector2.toArray
      } else {
        val tmpVector3 = for (i <- 0 until numOfSamples) yield uf.sample()
        tmpVector3.toArray
      }
    choiceVector
  }

  def kernelMatrix( kernelFunction: MercerKernel,
                    data: org.apache.spark.ml.linalg.DenseMatrix,
                   center: Boolean = false
                  ): org.apache.spark.ml.linalg.DenseMatrix = {

    val nrow = data.numRows
    val ncol = data.numCols
    val rowIter1 = data.rowIter
    val tmpArray = org.apache.spark.ml.linalg.Vectors.zeros(nrow * nrow).toArray
    var i: Int = 0
    while (rowIter1.hasNext) {
      val tmp1Vector = rowIter1.next.toDense
      var j: Int = 0
      val rowIter2 = data.rowIter
      while (rowIter2.hasNext && j <= i) {
        val tmp2Vector = rowIter2.next.toDense
        val tmp1Double = kernelFunction.k(tmp1Vector, tmp2Vector)
        tmpArray(i * nrow + j) = tmp1Double
        tmpArray(j * nrow + i) = tmp1Double
        j += 1
      }
      i += 1
    }
    val tmpMatrix = new org.apache.spark.ml.linalg.DenseMatrix(nrow, nrow, tmpArray)
    if (center) myUtilForKPCA.centralizeKernelMatrix(tmpMatrix) else tmpMatrix
  }

  def kernelMatrix(kernelFunction: MercerKernel,
                   dataX: org.apache.spark.ml.linalg.DenseMatrix,
                   dataY: org.apache.spark.ml.linalg.DenseMatrix
                  ): org.apache.spark.ml.linalg.DenseMatrix = {
    val nrowX = dataX.numRows
    val ncolX = dataX.numCols
    val nrowY = dataY.numRows
    val ncolY = dataY.numCols
    val rowIterY = dataY.rowIter
    val tmpArray = org.apache.spark.ml.linalg.Vectors.zeros(nrowX * nrowY).toArray
    var i: Int = 0
    while (rowIterY.hasNext) {
      val tmp1Vector = rowIterY.next.toDense
      val rowIterX = dataX.rowIter
      while (rowIterX.hasNext) {
        val tmp2Vector = rowIterX.next.toDense
        val tmp1Double = kernelFunction.k(tmp1Vector, tmp2Vector)
        tmpArray(i) = tmp1Double
        i += 1
      }
    }
    new org.apache.spark.ml.linalg.DenseMatrix(nrowX, nrowY, tmpArray)
  }
}
