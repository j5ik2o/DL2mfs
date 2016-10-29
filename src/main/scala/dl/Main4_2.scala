package dl

import breeze.linalg._
import breeze.numerics._
import dl.MinstLoader._

import scala.util.Random

object Main4_2 extends App {
  // 4.2.1 2 乗和誤差
  {
    def meanSquaredError(y: DenseVector[Double], t: DenseVector[Double]) = {
      0.5 * sum(pow(y - t, 2))
    }

    def crossEntropyError(y: DenseVector[Double], t: DenseVector[Double]): Double = {
      y.data.zip(t.data).aggregate(0.0)({ case (r, (y, x)) =>
        println(s"r = $r")
        r - x * Math.log(y + 1e-7)
      }, _ + _)
    }

    val t = DenseVector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    val y1 = DenseVector(0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0)
    val y2 = DenseVector(0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0)

    val e1 = meanSquaredError(y1, t)
    val e2 = meanSquaredError(y2, t)
    println(e1)
    println(e2)

    val c1 = crossEntropyError(y1, t)
    val c2 = crossEntropyError(y2, t)
    println(c1)
    println(c2)

  }
  {
    val (trainData, _, testData, _) = loadMinst(flatten = false)
    val size = trainData.size
    println(s"size = $size")
    val result = for {_ <- 0 until 10} yield {
      val idx = Random.nextInt(size)
      trainData(idx)
    }
    //println(s"result = $result")
  }
  // 4.2.4 ［バッチ対応版］交差エントロピー誤差の実装`
  {
    def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
      // E = -1/N ΣnΣk (Tnk * log(Ynk))
      val n = y.rows
      y.data.zip(t.data).aggregate(0.0)({ case (r, (y, t)) =>
        println(s"r = $r")
        r - t * Math.log(y + 1e-7)
      }, _ + _)  / n
    }

    val t = DenseMatrix(
      (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    val y1 = DenseMatrix(
      (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    val y2 = DenseMatrix(
      (0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.05, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    val e1 = crossEntropyError(y1, t)
    val e2 = crossEntropyError(y2, t)
    println(e1)
    println(e2)

  }
}
