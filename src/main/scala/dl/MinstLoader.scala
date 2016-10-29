package dl

import java.nio.ByteBuffer

import breeze.linalg._
import breeze.numerics._
import better.files._
import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.mutable.ArrayBuffer

object MinstLoader {

  private def changeOneHotLabel(in: DenseVector[Int]) = {
    val result = DenseMatrix.zeros[Int](in.size, 10)
    result.foreachKey{ case (x, y) =>
      if (in(x) == y)
        result(x, y) = 1
    }
    result
  }

  def loadLabel(path: String, oneHotLabel: Boolean = false): DenseMatrix[Int] = {
    val f = File(path)
    val ch = f.newFileChannel
    val headerBuf = ByteBuffer.allocateDirect(8)
    ch.read(headerBuf)
    headerBuf.rewind()
    val _ = headerBuf.getInt()
    val num = headerBuf.getInt()
    val buf = ByteBuffer.allocateDirect(num)
    ch.read(buf)
    buf.rewind()
    val resultSeq: Seq[Int] = for(i <- 0 until num) yield buf.get() & 0xff
    val result = DenseVector(resultSeq: _*)
    if (oneHotLabel)
      changeOneHotLabel(result)
    else
      result.toDenseMatrix
  }

  def loadData(path: String, normalize: Boolean = false, flatten: Boolean = false): Seq[DenseMatrix[Double]] = {
    val f = File(path)
    val ch = f.newFileChannel
    val headerBuf = ByteBuffer.allocateDirect(16)
    ch.read(headerBuf)
    headerBuf.rewind()

    val _ = headerBuf.getInt()
    val num = headerBuf.getInt()
    val w = headerBuf.getInt()
    val h = headerBuf.getInt()

    val buf = ByteBuffer.allocateDirect(num * w * h)
    ch.read(buf)
    buf.rewind()

    val result = ArrayBuffer.empty[DenseMatrix[Double]]
    for (n <- 0 until num) {
      val image = DenseMatrix.zeros[Double](h, w)
      for (y <- 0 until h; x <- 0 until w) {
        image(x, y) = if (normalize) (buf.get() & 0xff).toDouble / 255 else (buf.get() & 0xff).toDouble
      }
      result += image
    }
    if (flatten) {
      result.map{_.reshape(h * w, 1)}.result()
    } else {
      result.result()
    }
  }

  def loadMinst(normalize:Boolean = false, oneHotLabel: Boolean = false, flatten: Boolean = false): (Seq[DenseMatrix[Double]], DenseMatrix[Int], Seq[DenseMatrix[Double]], DenseMatrix[Int]) = {
    val trainData = loadData("train-images-idx3-ubyte", normalize, flatten)
    val trainLabels = loadLabel("train-labels-idx1-ubyte", oneHotLabel)
    val testData = loadData("t10k-images-idx3-ubyte", normalize, flatten)
    val testLabels = loadLabel("t10k-labels-idx1-ubyte", oneHotLabel)
    (trainData, trainLabels, testData, testLabels)
  }
}
