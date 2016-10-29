package dl

import java.awt.{Graphics, Graphics2D}
import javax.swing.{JFrame, JPanel}

import breeze.linalg._
import breeze.plot._
import breeze.numerics._

object FunctionsPanel extends JPanel with App {
  val f = Figure()
  val p = f.subplot(0)

  val frame = new JFrame()
  frame.getContentPane.add(this)
  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.setBounds(10, 10, 300, 200)
  frame.setTitle("タイトル")
  frame.setVisible(true)

  override def paintComponent(g: Graphics): Unit = {
    val x: DenseVector[Double] = DenseVector.rangeD(-5.0d, 5.0d, 0.1d)
    val y1: DenseVector[Double] = Main3_2.stepFunction(x)
    val y2: DenseVector[Double] = sigmoid(x)
    val y3: DenseVector[Double] = Main3_2.maxFunction(x)
    p += plot(x, y1, '-')
    p += plot(x, y2, '.')
    p += plot(x, y3, '+')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
    f.drawPlots(g.asInstanceOf[Graphics2D])
  }
}

object Main3_2 extends App {
  def maxFunction(x: DenseVector[Double]): DenseVector[Double] = {
    x.map(e => if (e > 0) e else 0)
  }

  def stepFunction(x: DenseVector[Double]): DenseVector[Double] = {
    val y = x :> 0.0d
    y.map(e => if (e) 1.0d else 0.0d)
  }

}

