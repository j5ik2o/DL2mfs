package dl

import breeze.linalg._
import breeze.numerics._

object Common {

  def crossEntropyError(y: DenseMatrix[Double], t: DenseMatrix[Double]): Double = {
    // E = -1/N ΣnΣk (Tnk * log(Ynk))
    val n = y.rows
    y.data.zip(t.data).aggregate(0.0)({ case (r, (y, t)) =>
      r - t * Math.log(y + 1e-7)
    }, _ + _)  / n
  }

  def softmax(a: DenseMatrix[Double]): DenseMatrix[Double] = {
    val c = max(a)
    val expA = exp(a - c)
    val sumExpA = sum(expA)
    expA / sumExpA
  }
}
