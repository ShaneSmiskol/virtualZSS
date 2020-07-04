import ast
import os
import pickle
import numpy as np
from utils.tokenizer import tokenize, split_list
from utils.BASEDIR import BASEDIR
import matplotlib.pyplot as plt
import json

os.chdir(BASEDIR)


SAVE_PATH = 'model_data'



class DataProcessor:
  def __init__(self):
    # zss key has bad data, use angle_steers instead as ground truth
    self.csv_keys = ['angle_steers', 'shitty_angle', 'zss', 'output_steer', 'wheel_speeds.fl', 'wheel_speeds.fr', 'wheel_speeds.rl', 'wheel_speeds.rr']

  def start(self):
    self.load_data()
    self.plot_data()

  def plot_data(self):
    shitty_angle = [line['shitty_angle'] for line in self.driving_data]
    angle_steers = [line['angle_steers'] for line in self.driving_data]
    plt.plot(shitty_angle, label='shitty_angle')
    plt.plot(angle_steers, label='good angle')
    plt.legend()
    plt.show()

  def load_data(self):
    self.driving_data = []
    for data_file in os.listdir('data/'):
      if 'broken' in data_file:
        print('Skipping file: {}'.format(data_file))
        continue
      print('Loading: {}'.format(data_file))
      with open('data/{}'.format(data_file), 'r') as f:
        data = f.read().split('\n')

      for line in data:
        try:
          line = dict(zip(self.csv_keys, ast.literal_eval('[{}]'.format(line))))
          if len(line) == 0:
            continue
          self.driving_data.append(line)
        except:
          print('error with line: {}'.format(line))


if __name__ == '__main__':
  dp = DataProcessor()
  dp.start()
