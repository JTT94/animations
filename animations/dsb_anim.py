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
charts_scale=0.6
config.background_color = WHITE
config["background_color"] = WHITE


im_height = 10
start_left= 40
start_up = 15
increment = 5
tex_scale = 2
show_stride = 5
start_right= increment*19-start_left

class DSBScene(MovingCameraScene):

    
    def play_forward(self, path_template,n, start_up):
        path = path_template.format(0, n)
        first_im = ImageMobject(path).shift(RIGHT*start_right).shift(UP*(start_up))
        first_im.height = im_height
        
        #sample_0 = Tex(r'Sample $x_T \sim \pi_T$', color=BLACK).scale(tex_scale).shift(UP*(start_up+7)).shift(LEFT*start_left)
        #sample_f = Tex(r'Sample $x_0 \sim \pi_0$', color=BLACK).scale(tex_scale).shift(UP*(start_up+7)).shift(RIGHT*start_right)
        #backward_0 = Tex(r'Forward $x_{t+1} = F_{\theta}(t, x_{t}) + \sqrt{2 \gamma_{t}} z_{t}$', color=BLACK).scale(tex_scale).shift(UP*(start_up+7))
        #self.add(sample_o)#, backward_0)
        self.play(FadeIn(first_im))   

        backward = Group()
        for i in range(1,20):
            if i % show_stride == 0:
                path = path_template.format(i,n)
                im = ImageMobject(path).shift(RIGHT*(start_right-increment*i)).shift(UP*start_up)
                backward.add(im)
                im.height = im_height
                arrow =  Arrow(start=RIGHT*(start_right-increment*(i-show_stride)), 
                               end=RIGHT*(start_right-increment*(i+0*show_stride)), 
                               max_stroke_width_to_length_ratio=10,
                               color=RED).shift(UP*start_up)
                backward.add(arrow)
                if i == show_stride:
                    self.play(FadeIn(im), FadeIn(arrow))
                else:
                 self.play(FadeIn(im),prev_im.animate.set_opacity(0.3), FadeIn(arrow))   
                prev_im =im

        path = path_template.format(i,n)
        final_im = ImageMobject(path).shift(RIGHT*(start_right-increment*i)).shift(UP*start_up)
        arrow =  Arrow(start=RIGHT*(start_right-increment*(i-show_stride)),
                       end=RIGHT*(start_right-increment*(i+0*show_stride)), 
                       max_stroke_width_to_length_ratio=10,
                       color=RED).shift(UP*start_up)
        backward.add(arrow)
        final_im.height = im_height
        self.play(FadeIn(final_im), prev_im.animate.set_opacity(0.3), FadeIn(arrow))
        self.wait(3)
        #backward.add(final_im)
        backward.add(first_im)
        return backward, final_im

    def play_backward(self, path_template, n, start_up):
        path = path_template.format(0,n)
        first_im = ImageMobject(path).shift(LEFT*start_left).shift(UP*start_up)
        first_im.height = im_height
        
        #sample_0 = Tex(r'Sample $x_T \sim \pi_T$', color=BLACK).scale(tex_scale).shift(UP*(start_up+7)).shift(LEFT*start_left)
        #backward_0 = Tex(r'Backward $x_{t-1} = B_{\theta}(t, x_{t}) + \sqrt{2 \gamma_{t}} z_{t}$', color=BLACK).scale(tex_scale).shift(UP*(start_up+7))
        #self.add(sample_0)#, backward_0)
        self.play(FadeIn(first_im))   

        backward = Group()
        for i in range(1,20):
            if i % show_stride == 0:
                path = path_template.format(i,n)
                im = ImageMobject(path).shift(LEFT*(start_left-increment*i)).shift(UP*start_up)
                backward.add(im)
                im.height = im_height
                arrow =  Arrow(start=LEFT*(start_left-increment*(i-show_stride)), 
                               end=LEFT*(start_left-increment*(i+0*show_stride)), 
                               max_stroke_width_to_length_ratio=10,
                               color=RED).shift(UP*start_up)
                backward.add(arrow)
                if i == show_stride:
                    self.play(FadeIn(im), FadeIn(arrow))
                else:
                 self.play(FadeIn(im),prev_im.animate.set_opacity(0.3), FadeIn(arrow))   
                prev_im =im

        path = path_template.format(i,n)
        final_im = ImageMobject(path).shift(LEFT*(start_left-increment*i)).shift(UP*start_up)
        final_im.height = im_height
        arrow =  Arrow(start=LEFT*(start_left-increment*(i-show_stride)), 
                       end=LEFT*(start_left-increment*(i+0*show_stride)), 
                       max_stroke_width_to_length_ratio=10.,
                       color=RED).shift(UP*start_up)
        backward.add(arrow)
        self.play(FadeIn(final_im), prev_im.animate.set_opacity(0.3), FadeIn(arrow))
        self.wait(3)
        #backward.add(final_im)
        backward.add(first_im)
        return backward, final_im

    def construct(self):

        self.camera.frame.set(width=120)
        backward_template = '/data/hylia/thornton/animations/mnist_im/backward_{1}_5000/im_grid_{0}.png'
        forward_template = '/data/hylia/thornton/animations/mnist_im/forward_{1}_5000/im_grid_{0}.png'
        forward_0_template = '/data/hylia/thornton/animations/mnist_im/forward_{1}_0/im_grid_{0}.png'

        start_up = 25
        f0_text = Text("Forward 0", font_size=144,color=BLACK).shift(UP*(start_up+7))#.shift(RIGHT*start_right)
        self.add(f0_text)
        forward_0, f0 = self.play_forward(forward_0_template, 0, start_up)

        start_up=10
        b1_text = Text("Backward 1, reverse forward 0", font_size=144,color=BLACK).shift(UP*(start_up+7))#.shift(RIGHT*start_right)
        self.add(b1_text)
        backward_1, b1 = self.play_backward(backward_template, 1, start_up)

        start_up=-5
        fN_text = Text("Forward N-1, reverse backward N-1", font_size=144,color=BLACK).shift(UP*(start_up+7))#.shift(RIGHT*start_right)
        self.add(fN_text)
        forward_N, fN = self.play_forward(forward_template, 15, start_up)

        start_up=-20
        bN_text = Text("Backward N, reverse forward N-1", font_size=144,color=BLACK).shift(UP*(start_up+7))#.shift(RIGHT*start_right)
        self.add(bN_text)
        backward_N, bN = self.play_backward(backward_template, 16, start_up)
        
        self.play(FadeOut(forward_0), 
                  FadeOut(backward_1), 
                  FadeOut(forward_N), 
                  FadeOut(backward_N),
                  FadeOut(bN_text),
                  FadeOut(fN_text),
                  FadeOut(b1_text),
                  FadeOut(f0_text), 
                  FadeOut(f0),
                  FadeOut(fN))

     
        self.play(bN.animate.move_to(15*LEFT), b1.animate.move_to(15*RIGHT))
        self.play(self.camera.frame.animate.set(width=50))
        self.wait(15)
        
        
        
        

    
