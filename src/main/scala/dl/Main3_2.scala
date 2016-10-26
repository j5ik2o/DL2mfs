package dl

import breeze.linalg._
import breeze.plot._
import breeze.numerics._

object Main3_2 extends App {
  def maxFunction(x: DenseVector[Double]): DenseVector[Double] = {
    x.map(e => if(e > 0) e else 0)
  }
  def stepFunction(x: DenseVector[Double]): DenseVector[Double] = {
    val y = x :> 0.0d
    y.map(e => if (e) 1.0d else 0.0d)
  }

  //val m = DenseMatrix.zeros[Double](5, 1)

  val x: DenseVector[Double] = DenseVector.rangeD(-5.0d, 5.0d, 0.1d)
  val y1: DenseVector[Double] = stepFunction(x)
  val y2: DenseVector[Double] = sigmoid(x)
  val y3: DenseVector[Double] = maxFunction(x)

  val f = Figure()
  val p = f.subplot(0)
  p += plot(x, y1, '-')
  p += plot(x, y2, '.')
  p += plot(x, y3, '+')
  p.xlabel = "x axis"
  p.ylabel = "y axis"
  f.saveas("lines.png")



}

