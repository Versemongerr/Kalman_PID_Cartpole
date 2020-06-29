# Inspired by https://keon.io/deep-q-learning/

import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.classic_control import *
from cartpole_env import *

import matplotlib.patches as mpatches

kp_cart = 2.4 - 0.5 + 0.1 + 0.25
kd_cart = 70 + 5 + 5 - 5 - 1.5
ki_cart = 0.008 + 0.001 + 0.004

kp_pole = 8 - 0.5
kd_pole = 100 - 5
ki_pole = 0.005

DIRECT_MAG = True
RANDOM_NOISE = False

k1 = -1/(0.5*(4/3-0.1/1.1)*1.1)
k2 = 1/1.1 - 0.1*0.5*k1/1.1

z_mat = np.mat([[0.0], [0.0]])
x_mat = np.mat([[0.0], [0.0], [0.0], [0.0]])
p_mat = np.mat([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

# f_mat = np.mat([[1, 0.02, 0, 0], [-9.8*k1, 1.0, 0, 0], [0, 0, 1, 0.02], [0.0, 0, 0, 1]])
f_mat = np.mat([[1, 0.02, 0, 0], [0, 1.0, 0, 0], [0, 0, 1, 0.02], [0.0, 0, 0, 1]])

b_mat = np.mat([[0], [0.02*k1], [0], [0.02*k2]])
q_mat = np.mat([[0.0001, 0, 0, 0], [0, 0.0001, 0, 0], [0, 0, 0.0001, 0], [0, 0, 0, 0.0001]])
h_mat = np.mat([[0, 1.0, 0, 0], [0, 0, 0, 1.0]])
r_mat = np.mat([[0.25, 0], [0, 0.25]])

err = [0.0, 0.0, 0.0, 0.0]
if DIRECT_MAG:
    env = CartPoleEnv()
else:
    env = gym.make('CartPole-v1')


class KFilter:
    def __init__(self, f_mat, b_mat, q_mat, h_mat, r_mat):
        self.f_mat = f_mat
        self.b_mat = b_mat
        self.q_mat = q_mat
        self.h_mat = h_mat
        self.r_mat = r_mat

    def kal_filter(self, x_mat, p_mat, z_mat, action):
        x_predict = self.f_mat * x_mat + self.b_mat * action
        p_predict = self.f_mat * p_mat * self.f_mat.T + self.q_mat
        k_num = p_predict * self.h_mat.T * np.linalg.pinv(self.h_mat * p_predict * self.h_mat.T + self.r_mat)
        x_mat = x_predict + k_num * (z_mat - self.h_mat * x_predict)
        # print(x_predict[0])
        p_mat = (np.eye(4) - k_num * self.h_mat) * p_predict
        return x_mat, p_mat


class CartPoleControl:

    def __init__(self, kp_cart, ki_cart, kd_cart, kp_pole, ki_pole, kd_pole):
        self.kp_cart = kp_cart
        self.kd_cart = kd_cart
        self.ki_cart = ki_cart

        self.kp_pole = kp_pole
        self.kd_pole = kd_pole
        self.ki_pole = ki_pole

        self.bias_cart_1 = 0
        self.bias_pole_1 = 0

        self.pole_int = 0
        self.cart_int = 0
        self.i = 0

    def pid_cart(self, position):
        bias = position  # 这句可能有问题
        # bias=self.bias_cart_1*0.8+bias*0.2
        d_bias = bias - self.bias_cart_1
        self.cart_int += bias
        balance = self.kp_cart * bias + self.kd_cart * d_bias + self.ki_cart * self.cart_int
        self.bias_cart_1 = bias
        return balance

    def pid_pole(self, angle):
        bias = angle  # 这句可能有问题
        d_bias = bias - self.bias_pole_1
        self.pole_int += bias
        balance = -self.kp_pole * bias - self.kd_pole * d_bias - self.ki_pole * self.pole_int
        self.bias_pole_1 = bias
        return balance

    def control_output(self, control_cart, control_pole):
        if DIRECT_MAG:
            return -10*(control_pole - control_cart)
        else:
            return 1 if (control_pole - control_cart) < 0 else 0


if __name__ == '__main__':

    control = CartPoleControl(kp_cart, ki_cart, kd_cart, kp_pole, ki_pole, kd_pole)

    kf = KFilter(f_mat, b_mat, q_mat, h_mat, r_mat)

    rewards = 0
    state = env.reset()
    x_mat[0][0] = state[2]
    x_mat[1][0] = state[3]
    x_mat[2][0] = state[0]
    x_mat[3][0] = state[1]
    noisy_state = state
    # print(x_mat)
    # print(state)
    done = False
    i = 0
    j = 0
    figure, ax = plt.subplots(2, 2)
    while (j < 1000) & (abs(state[2] < 2)):
    # while abs(state[2] < 2):
        j = j + 1
        env.render()
        # control_pole = control.pid_pole(state[2])
        # control_cart = control.pid_cart(state[0])
        control_pole = control.pid_pole(x_mat[0, 0])
        control_cart = control.pid_cart(x_mat[2, 0])
        # control_pole = control.pid_pole(noisy_state[2])
        # control_cart = control.pid_cart(noisy_state[0])

        if RANDOM_NOISE and random.random() > 0.99:
            i = 2

        if i > 0:
            if DIRECT_MAG:
                action = 10
            else:
                action = 1
            i -= 1
        else:
            action = control.control_output(control_cart, control_pole)
        # action = 0

        next_state, reward, done, _ = env.step(action)

        noise = np.random.normal(loc=0, scale=0.5, size=4)
        noisy_state = next_state + noise
        z_mat[0][0] = noisy_state[3]
        z_mat[1][0] = noisy_state[1]
        # x_predict = f_mat * x_mat + b_mat * action
        # # print(x_predict)
        # p_predict = f_mat * p_mat * f_mat.T + q_mat
        # k_num = p_predict * h_mat.T * np.linalg.pinv(h_mat * p_predict * h_mat.T + r_mat)
        # x_mat = x_predict + k_num * (z_mat - h_mat * x_predict)
        # p_mat = (np.eye(4) - k_num * h_mat) * p_predict
        # print(x_mat.T)
        x_mat, p_mat = kf.kal_filter(x_mat, p_mat, z_mat, action)

        state = next_state
        rewards += reward
        # print(state)
        # print(action)
        ax[0][0].plot(j, x_mat[2, 0], 'ro', markersize=1)
        ax[0][0].plot(j, state[0], 'go', markersize=1)
        ax[0][0].plot(j, noisy_state[0], 'bo', markersize=1)

        ax[0][1].plot(j, x_mat[3, 0], 'ro', markersize=1)
        ax[0][1].plot(j, state[1], 'go', markersize=1)
        ax[0][1].plot(j, noisy_state[1], 'bo', markersize=1)

        ax[1][0].plot(j, x_mat[0, 0], 'ro', markersize=1)
        ax[1][0].plot(j, state[2], 'go', markersize=1)
        ax[1][0].plot(j, noisy_state[2], 'bo', markersize=1)

        ax[1][1].plot(j, x_mat[1, 0], 'ro', markersize=1)
        ax[1][1].plot(j, state[3], 'go', markersize=1)
        ax[1][1].plot(j, noisy_state[3], 'bo', markersize=1)
    print('total rewards:'+str(rewards))
    env.close()

    color = ['red', 'blue', 'green']
    labels = ['kalman filtered position', 'only use measured position', 'truth position']
    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]

    ax[0][0].set_title('x')
    ax[0][0].set_xlabel('step')
    ax[0][0].set_ylabel('x')
    ax[0][0].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)  # 生成legend

    ax[0][1].set_title('x_dot')
    ax[0][1].set_xlabel('step')
    ax[0][1].set_ylabel('x/s')
    ax[0][1].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)  # 生成legend

    ax[1][0].set_title('theta')
    ax[1][0].set_xlabel('step')
    ax[1][0].set_ylabel('rad')
    ax[1][0].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)  # 生成legend

    ax[1][1].set_title('theta_dot')
    ax[1][1].set_xlabel('step')
    ax[1][1].set_ylabel('rad/s')
    ax[1][1].legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)  # 生成legend

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.legend(handles=patches, bbox_to_anchor=(0, 1), loc=2, borderaxespad=0)  # 生成legend
    plt.show()
