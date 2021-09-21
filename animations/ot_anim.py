from manim import *
import numpy as np
import itertools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ott

from ott.geometry import pointcloud
from ott.core import sinkhorn
from ott.tools import transport



class BarChartExample(Scene):


    def construct(self):

        # ot
        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 3)

        n, m, d = 10, 10, 2
        x = jax.random.normal(rngs[0], (n,d)) + 1
        y = x# jax.random.uniform(rngs[1], (m,d))
        x = jnp.sort(x)


        a = jax.random.uniform(rngs[0], (n,))
        b = jax.random.uniform(rngs[1], (m,))
        a = a / jnp.sum(a)
        b = b / jnp.sum(b)

        geom = pointcloud.PointCloud(x, y, epsilon=1e-2)
        out = sinkhorn.sinkhorn(geom, a, b)
        P = geom.transport_from_potentials(out.f, out.g)        
        P = P / P.max()

        
        # hist
        left_bar = BarChart(
            a,
            max_value=max(a),
            bar_colors=["#f72c01"],
            bar_names= ["x_{0}".format(i) for i in range(len(x))],
            bar_label_scale_val=0.3,
        ).scale(0.4).move_to(LEFT*3).shift(1.5*DOWN)
        self.add(left_bar)


        right_bar = BarChart(
            b,
            max_value=max(b),
            bar_colors=['#0162f7'],
            bar_names=["y_{0}".format(i) for i in range(len(x))],
            bar_label_scale_val=0.3,
        ).scale(0.4).move_to(RIGHT*3).shift(1.5*DOWN)

        self.add(right_bar)

        left_tex = Tex(r'Given $\alpha=\sum_{i=1}^n a_i \delta_{x_i}$, $\beta=\sum_{i=1}^m b_i \delta_{y_i}$').scale(0.5).move_to(LEFT*3).shift(3*UP)
        self.add(left_tex)

        ot_tex = Tex(r'Find $p^*_{i.j} = \arg\min_{p_{i.j}} \sum_{i,j} c(x_i,y_j)p_{i,j} + \epsilon p_{i,j} \log \frac{p_{i,j}}{a_ib_j}$').scale(0.5).shift(2.5*UP).shift(0.5*LEFT)
        self.add(ot_tex)

        marginal_1 = Tex(r'subject to $\sum_i p_{i,j} = b_j$, $\sum_j p_{i,j} = a_i$').scale(0.5).shift(2*UP).shift(LEFT)
        self.add(marginal_1)    
        

        #right_bar.rotate(- PI / 2)
        right_bar.generate_target()
        right_bar.target.shift(2*UP).shift(4*LEFT)
        #self.play()

        left_bar.generate_target()
        left_bar.target.shift(1*RIGHT).shift(4*DOWN)
        #self.play(MoveToTarget(left_bar))

        self.play(left_bar.animate.flip(UP)) # FadeOut(right_tex), FadeOut(left_tex)
        self.play(MoveToTarget(right_bar), MoveToTarget(left_bar),
                  left_bar.animate.rotate(90*DEGREES))
        self.play(FadeOut(left_bar.y_axis), FadeOut(right_bar.y_axis), FadeOut(right_bar.y_axis_labels), FadeOut(left_bar.y_axis_labels))

        #self.play(Create(m))
        

        ys = []
        xs = []
        for bar in left_bar.bars:
            y = bar.get_bottom()
            ys.append(y[1])

            #tex = Tex(r'x').move_to(bar_bottom, DOWN)
            #self.add(tex)

        for bar in right_bar.bars:
            x = bar.get_bottom()
            xs.append(x[0])

        coordinates = np.array(list(itertools.product(xs, ys)))
        coordinates = np.column_stack((coordinates, np.zeros(coordinates.shape[0])))

        i=0
        P = P.flatten()
        points = VGroup()
        for c in coordinates:
            point = Circle(radius=P[i]).scale(0.05).set_fill(RED, opacity=0.8).move_to(c)
            i+=1
            points.add(point)
        self.play(FadeIn(points))

        max_x = np.max(coordinates[:,0])
        min_x = np.min(coordinates[:,0])
        max_y = np.max(coordinates[:,1])
        min_y = np.min(coordinates[:,1])
        p1=[max_x, min_y,0.]
        p2=[max_x, max_y,0.]
        brace = BraceBetweenPoints(p1,p2)
        self.play(Create(brace))
        t = Tex(r"$p^*_{i,j}$").next_to(brace, RIGHT)
        self.add(t)

        self.wait(3)

        