package dl

import javax.swing.{JFrame, JPanel}
import java.awt.{Graphics, Graphics2D}

import breeze.linalg._
import breeze.plot._
import breeze.numerics._

object Function1Panel extends JPanel with App {
  val f = Figure()
  val p = f.subplot(0)

  val frame = new JFrame()
  frame.getContentPane.add(this)
  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.setBounds(10, 10, 300, 200)
  frame.setTitle("タイトル")
  frame.setVisible(true)

  override def paintComponent(g: Graphics): Unit = {
    val x: DenseVector[Double] = DenseVector.rangeD(0.0d, 20.0d, 0.1d)
    val y1: DenseVector[Double] = x.map(Main4_3.function1)
    val y2: DenseVector[Double] = x.map(Main4_3.tangentLine(Main4_3.function1, 5))
    p += plot(x, y1)
    p += plot(x, y2)
    p.title = "y = function1(x)"
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    f.drawPlots(g.asInstanceOf[Graphics2D])
  }
}

object Function2Panel extends JPanel with App {
  val f = Figure()
  val p = f.subplot(0)

  val frame = new JFrame()
  frame.getContentPane.add(this)
  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.setBounds(10, 10, 300, 200)
  frame.setTitle("タイトル")
  frame.setVisible(true)

  override def paintComponent(g: Graphics): Unit = {
    val x = DenseVector.rangeD(-2.0d, 2.5d + 0.25d, 0.25d)
    val r1 = (1 until x.activeSize).foldLeft(x.toDenseMatrix) { (r, _) => DenseMatrix.vertcat(r, x.toDenseMatrix) }.flatten()
    val r2 = (1 until x.activeSize).foldLeft(x.toDenseMatrix.reshape(x.size, 1)) { (r, _) => DenseMatrix.horzcat(r, x.toDenseMatrix.reshape(x.size, 1)) }.flatten()

    //    val x0: DenseVector[Double] = DenseVector.rangeD(-2.0d, 2.5d, 0.25d)
    //    val x1: DenseVector[Double] = DenseVector.rangeD(-2.0d, 2.5d, 0.25d)
    //    x0.toDenseMatrix * 2
    //
    //
    //    val y1: DenseVector[Double] = x.map(Main4_3.function1)
    //    val y2: DenseVector[Double] = x.map(Main4_3.tangentLine(Main4_3.function1, 5))
    //    p += plot(x, y1)
    //    p += plot(x, y2)
    //    p.title = "y = function2(x)"
    //    p.xlabel = "x axis"
    //    p.ylabel = "y axis"
    //    f.drawPlots(g.asInstanceOf[Graphics2D])
  }

}

object Main4_3 extends App {
  // 4.4.1 勾配法
  def function1(x: Double): Double = {
    0.01 * Math.pow(x, 2) + 0.1 * x
  }

  def tangentLine(f: (Double) => Double, x: Double): (Double) => Double = {
    val d = numericalDiff(f, x)
    val y = f(x) - d * x
    (t: Double) => {
      d * t + y
    }
  }

  def numericalDiff(f: (Double) => Double, x: Double): Double = {
    val h: Double = 1e-4
    (f(x + h) - f(x - h)) / (2 * h)
  }

  def function2(x: DenseVector[Double]): Double = {
    sum(pow(x, 2))
  }

  def function2_2(x: DenseMatrix[Double]): Double = {
    sum(pow(x, 2))
  }

  // 4.3.2 数値微分の例
  {
    println(numericalDiff(function1, 5))
    println(numericalDiff(function1, 10))
  }
  // 4.3.3 偏微分
  {
    def functionTmpl1(x0: Double) =
      x0 * x0 + Math.pow(4.0, 2.0)
    def functionTmpl2(x1: Double) =
      Math.pow(3.0, 2.0) + x1 * x1

    println(numericalDiff(functionTmpl1, 3.0))
    println(numericalDiff(functionTmpl2, 4.0))

  }

  def numericalGradient(f: (DenseVector[Double]) => Double, x: DenseVector[Double]): DenseVector[Double] = {
    val h = 1e-4
    val result = for {idx <- 0 until x.size} yield {
      val tmp = x(idx)
      x(idx) = x(idx) + h
      val fxh1 = f(x)
      x(idx) = tmp - h
      val fxh2 = f(x)
      val r = (fxh1 - fxh2) / (2 * h)
      x(idx) = tmp
      r
    }
    DenseVector(result: _*)
  }

  def numericalGradient2(f: (DenseMatrix[Double]) => Double, x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val h = 1e-4
    val grad = DenseMatrix.zeros[Double](x.rows, x.cols)
    x.keysIterator.foreach { case (px, py) =>
      val tmp = x(px, py)
      x(px, py) = x(px, py) + h
      val fxh1 = f(x)
      x(px, py) = tmp - h
      val fxh2 = f(x)
      val r = (fxh1 - fxh2) / (2 * h)
      x(px, py) = tmp
      grad(px, py) = r
    }
    grad
  }

  {
    val r = numericalGradient(function2, DenseVector(3.0, 4.0))
    println(r)
    val r2 = numericalGradient2(function2_2, DenseMatrix(3.0, 4.0))
    println(r2)
  }

  def gradientDescent(f: (DenseVector[Double]) => Double, initX: DenseVector[Double], lr: Double = 0.01, stepNum: Int = 100) = {
    var x = initX.copy
    for (i <- 0 until stepNum) {
      val grad = numericalGradient(f, x)
      x -= lr * grad
    }
    x
  }
  {
    val x = DenseVector(-3.0, 4.0)
    val r = gradientDescent(function2, x, 0.1, 100)
    println(x, r)
  }
  // 学習率が大きすぎる例：lr=10.0
  {
    val x = DenseVector(-3.0, 4.0)
    val r2 = gradientDescent(function2, x, 10.0, 100)
    println(x, r2)
  }
  // 学習率が小さすぎる例：lr=1e-10
  {
    val x = DenseVector(-3.0, 4.0)
    val r = gradientDescent(function2, x, 1e-10, 100)
    println(x, r)
  }
  // 4.4.2 ニューラルネットワークに対する勾配
  {

    import breeze.stats.distributions.Gaussian
    val g = Gaussian.distribution(0.0, 1.0)

    def row = g.samplesVector(3).toDenseMatrix

    val w: DenseMatrix[Double] = (0 until 1).foldLeft(row) { (r, _) => DenseMatrix.vertcat(r, row) }
    println(w, w.rows, w.cols)

    def predict(x: DenseVector[Double]) = {
      x.toDenseMatrix * w
    }

    def loss(x: DenseVector[Double], t: DenseMatrix[Double]) = {
      val z = predict(x)
      val y = Common.softmax(z)
      val loss = Common.crossEntropyError(y, t)
      loss
    }

    val x = DenseVector(0.6, 0.9)
    val p = predict(x)
    println(p)

    println(argmax(p))

    val t = DenseVector(0.0, 0.0, 1.0).toDenseMatrix
    val r = loss(x, t)
    println(r)

    def f(v: DenseMatrix[Double]) = loss(x, t)
    val dW = numericalGradient2(f, w)
    println(dW)
  }
}
