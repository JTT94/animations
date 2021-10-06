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



def set_background(self):
    background = Rectangle(
    width = FRAME_WIDTH,
    height = FRAME_HEIGHT,
    stroke_width = 0,
    fill_color = "#3E746F",
    fill_opacity = 1)
    self.add(background)


red = "#f72c01"
blue = '#0162f7'
tex_scale = 0.8
charts_down = 0.
class OTScene(Scene):


    def construct(self):
        #set_background(self)

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
            bar_colors=[red],
            bar_names= ["x_{0}".format(i) for i in range(len(x))],
            bar_label_scale_val=0.3,
        ).scale(0.4).move_to(LEFT*3).shift(charts_down*DOWN)
        self.add(left_bar)


        right_bar = BarChart(
            b,
            max_value=max(b),
            bar_colors=[blue],
            bar_names=["y_{0}".format(i) for i in range(len(x))],
            bar_label_scale_val=0.3,
        ).scale(0.4).move_to(RIGHT*3).shift(charts_down*DOWN)

        self.add(right_bar)
        # left_tex = Tex(r'Given ', r'$\alpha=\sum_{i=1}^n a_i \delta_{x_i}$',', ', r'$\beta=\sum_{i=1}^m b_i \delta_{y_i}$').scale(tex_scale).move_to(LEFT*3).shift(3*UP)
        # left_tex.set_color_by_tex(r"$\alpha=\sum_{i=1}^n a_i \delta_{x_i}$", color=red, substring=False)
        # left_tex.set_color_by_tex(r'$\beta=\sum_{i=1}^m b_i \delta_{y_i}$', color=blue, substring=False)
        # self.add(left_tex)

        # ot_tex = Tex(r'Find $p^*_{i.j} = \arg\min_{p_{i.j}} \sum_{i,j} c(x_i,y_j)p_{i,j} + \epsilon p_{i,j} \log \frac{p_{i,j}}{a_ib_j}$').scale(tex_scale).shift(2.5*UP).shift(0.5*LEFT)
        # self.add(ot_tex)

        # marginal_1 = Tex(r'subject to $\sum_i p_{i,j} = b_j$, $\sum_j p_{i,j} = a_i$').scale(tex_scale).shift(2*UP).shift(LEFT)
        # self.add(marginal_1)    
        
        alpha_tex = Tex(r'$\alpha$', color=red).next_to(left_bar, DOWN)
        beta_tex = Tex(r'$\beta$', color=blue).next_to(right_bar, DOWN)
        self.add(alpha_tex)
        self.add(beta_tex)

        #right_bar.rotate(- PI / 2)
        right_bar.generate_target()
        right_bar.target.shift(2*UP).shift(4*LEFT)
        #self.play()

        left_bar.generate_target()
        left_bar.target.shift(1*RIGHT).shift(4*DOWN)
        #self.play(MoveToTarget(left_bar))

        self.play(left_bar.animate.flip(UP),FadeOut(alpha_tex), FadeOut(beta_tex)) # FadeOut(right_tex), FadeOut(left_tex)
        self.play(MoveToTarget(right_bar), MoveToTarget(left_bar),
                  left_bar.animate.rotate(90*DEGREES))
        
        alpha_tex = Tex(r'$\alpha$', color=red).next_to(left_bar, DOWN)
        beta_tex = Tex(r'$\beta$', color=blue).next_to(right_bar, RIGHT)
        self.play(FadeOut(left_bar.y_axis), FadeOut(right_bar.y_axis), 
                  FadeOut(right_bar.y_axis_labels), FadeOut(left_bar.y_axis_labels),
                  FadeIn(alpha_tex), FadeIn(beta_tex))

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
            point = Circle(radius=P[i], color=WHITE).scale(0.05).set_fill(WHITE, opacity=0.8).move_to(c)
            i+=1
            points.add(point)

        # brace

        max_x = np.max(xs)
        min_x = np.min(xs)
        max_y = np.max(ys)
        min_y = np.min(ys)

        p1=[min_x, min_y,0.]
        p2=[max_x, min_y,0.]
        brace = BraceBetweenPoints(p1,p2)

        t = Tex(r"$p^*_{i,j}$").next_to(brace, DOWN).shift(0.2*UP)
        self.play(FadeIn(points), FadeIn(brace), FadeIn(t))
    

        # bounding box
        xs.sort()
        ys.sort()

        k=5
        l_x = xs[k] - (xs[k]-xs[k-1])/2
        r_x = xs[k] + (xs[k+1]-xs[k])/2
        
        box_v = Polygon([l_x, min_y, 0], [r_x, min_y, 0], [r_x, max_y, 0], [l_x, max_y, 0], color=blue)
        v_sum = Tex(r"$\sum_ip^*_{i,j}=b_j$",color=blue).next_to(box_v, RIGHT).shift(0.3*RIGHT)
        self.play(FadeIn(box_v), FadeIn(v_sum))
        self.play(FadeOut(box_v), FadeOut(v_sum))
        

        b_y = ys[k] - (ys[k]-ys[k-1])/2
        t_y = ys[k] + (ys[k+1]-ys[k])/2
        box_h= Polygon([min_x, b_y, 0], [max_x, b_y, 0], [max_x, t_y, 0], [min_x, t_y, 0], color=red)
        h_sum = Tex(r"$\sum_jp^*_{i,j}=a_j$",color=red).next_to(box_h, RIGHT).shift(0.3*RIGHT)
        self.play(FadeIn(box_h), FadeIn(h_sum))


        
        self.wait(3)

        