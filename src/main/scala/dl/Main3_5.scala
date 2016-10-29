package dl

import java.awt.image.BufferedImage
import java.io.{File => JFile}
import javax.imageio.ImageIO

import better.files._
import breeze.linalg._
import breeze.numerics._
import dl.MinstLoader._

object Main3_5 extends App {
  {
    val a = DenseVector(0.3, 2.9, 4.0)
    val expA = exp(a)
    val sumExpA = sum(expA)
    val y = expA / sumExpA
    println(y)
  }
  // softmax
  {
    val a = DenseVector(0.3, 2.9, 4.0)
    val c = max(a)
    val expA = exp(a - c)
    val sumExpA = sum(expA)
    val y = expA / sumExpA
    println(y)
  }
  {
    def createImageFiles(prefix: String, data: Seq[DenseMatrix[Double]]) = {
      for {i <- 0 until 10} {
        val targetFile = File(s"$prefix-$i.jpeg")
        if (targetFile.exists) {
          println(s"The target file will be delete: file = $targetFile")
          targetFile.delete()
        }
        val fd = data(i)
        println(s"majorStride = ${fd.majorStride}")
        println(s"activeSize = ${fd.activeSize}")

        val img = new BufferedImage(fd.majorStride, fd.majorStride, BufferedImage.TYPE_BYTE_GRAY)
        img.getRaster.setPixels(0, 0, fd.majorStride, fd.majorStride, fd.data)
        ImageIO.write(img, "jpeg", targetFile.toJava)
      }
    }
    val (trainData, _, testData, _) = loadMinst(flatten = false)
    createImageFiles("train", trainData)
    createImageFiles("test", testData)
  }

}
