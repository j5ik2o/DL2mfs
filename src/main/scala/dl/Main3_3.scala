package dl

import breeze.linalg._
import breeze.numerics._

object Main3_3 extends App {

  implicit def toRichDenseVector[A](denseVector: DenseVector[A]) = new {
    def shape: Int = {
      denseVector.length
    }
  }

  implicit def toRichDenseMatrix[A](denseMatrix: DenseMatrix[A]) = new {
    def shape: (Int, Int) = {
      (denseMatrix.rows, denseMatrix.cols)
    }
  }

  // 3.3.1 多次元配列
  {
    println("---")
    val a = DenseVector(1.0, 2.0, 3.0, 4.0) // np.array([1, 2, 3, 4])
    println(s"a = $a")
    println(s"a.stride = ${a.stride}") // np.ndim(a)
    println(s"a.shape = ${a.shape}") // np.shape or np.shape[0]
  }
  {
    println("---")
    val b: DenseMatrix[Double] = DenseMatrix((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    println(s"b = $b")
    println(s"b.majorStride = ${b.majorStride}") // np.ndim(b)
    println(s"b.shape = ${b.shape}")
  }
  // 3.3.2 行列の内積
  {
    println("---")
    val a: DenseMatrix[Int] = DenseMatrix((1, 2), (3, 4))
    println(s"a.rows, a.cols = ${(a.rows, a.cols)}")
    val b: DenseMatrix[Int] = DenseMatrix((5, 6), (7, 8))
    println(s"b.shape = ${b.shape}")
    val r: DenseMatrix[Int] = a * b
    println(s"r = $r")
  }
  {
    println("---")
    val a: DenseMatrix[Double] = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    println(s"a = $a")
    println(s"a.shape = ${a.shape}")
    val r = a.reshape(1, a.size)
    println(s"reshape = $r")
    val r2 = DenseMatrix.vertcat(r, r).reshape(1, r.size * 2, View.Require)
    println(s"vertcat = $r2")

    val b: DenseMatrix[Double] = DenseMatrix((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    println(s"b.shape = ${b.shape}")
    val c: DenseMatrix[Double] = a * b
    println(s"c = $c")
  }
  {
    println("---")
    val a = DenseMatrix((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    println(s"a.shape = ${a.shape}")
    val b = DenseVector(7.0, 8.0)
    println(s"b.shape = ${b.shape}")
    val c = a * b
    println(s"c = $c")
  }
  // 3.3.3 ニューラルネットワークの内積
  {
    println("---")
    val m1 = DenseMatrix((1.0, 2.0))
    println(s"m1.shape = ${m1.shape}")
    val m2 = DenseMatrix((1.0, 3.0, 5.0),(2.0, 4.0, 6.0))
    println(s"m2.shape = ${m2.shape}")
    val m3 = m1 * m2
    println(s"m3 = $m3")
  }
  // 3.4.2 各層における信号伝達の実装
  {
    println("---")

    def identityFunction[A](x: A) = x

    val x = DenseVector(1.0, 0.5)
    println(s"x.stride = ${x.stride}") // np.ndim(x)
    println(s"x.shape = ${x.shape}")
    val w1 = DenseMatrix((0.1, 0.3, 0.5), (0.2, 0.4, 0.6))
    println(s"w1.shape = ${w1.shape}")
    val b1 = DenseVector(0.1, 0.2, 0.3)
    println(s"b1.shape = ${b1.shape}")

    val a1 = x.toDenseMatrix * w1 + b1.toDenseMatrix
    println(s"a1 = $a1")
    val z1 = sigmoid(a1).toDenseVector
    println(s"z1 = $z1")
    println(s"z1.shape = ${z1.shape}")

    val w2 = DenseMatrix((0.1, 0.4), (0.2, 0.5), (0.3, 0.6))
    println(s"w2.shape = ${w2.shape}")
    val b2 = DenseVector(0.1, 0.2)
    println(s"b2.shape = ${b2.shape}")

    val a2 = z1.toDenseMatrix * w2 + b2.toDenseMatrix
    println(s"a2 = $a2")
    val z2 = sigmoid(a2).toDenseVector
    println(s"z2 = $z2")

    val w3 = DenseMatrix((0.1, 0.3), (0.2, 0.4))
    val b3 = DenseVector(0.1, 0.2)

    val a3 = z2.toDenseMatrix * w3 + b3.toDenseMatrix
    println(s"a3 = $a3")

    val y = identityFunction(a3)
    println(s"y = $y")
  }
  // 3.4.3 実装のまとめ
  {
    def identityFunction[A](x: A) = x
    def initNetwork() = {
      val network: collection.mutable.Map[String, DenseMatrix[Double]] = collection.mutable.Map.empty[String, DenseMatrix[Double]]
      network("w1") = DenseMatrix((0.1, 0.3, 0.5), (0.2, 0.4, 0.6))
      network("b1") = DenseVector(0.1, 0.2, 0.3).toDenseMatrix
      network("w2") = DenseMatrix((0.1, 0.4), (0.2, 0.5), (0.3, 0.6))
      network("b2") = DenseVector(0.1, 0.2).toDenseMatrix
      network("w3") = DenseMatrix((0.1, 0.3), (0.2, 0.4))
      network("b3") = DenseVector(0.1, 0.2).toDenseMatrix
      network
    }

    def forward(network: collection.mutable.Map[String, DenseMatrix[Double]], x: DenseVector[Double]) = {
      val (w1, w2, w3) = (network("w1"), network("w2"), network("w3"))
      val (b1, b2, b3) = (network("b1"), network("b2"), network("b3"))

      val a1 = x.toDenseMatrix * w1 + b1
      val z1 = sigmoid(a1)
      val a2 = z1 * w2 + b2
      val z2 = sigmoid(a2)
      val a3 = z2 * w3 + b3
      val y = identityFunction(a3)
      y
    }

    println("---")
    val network = initNetwork()
    val x = DenseVector(1.0, 0.5)
    val y = forward(network, x)
    println(y)

  }

}
