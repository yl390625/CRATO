import numpy as np
import pandas as pd


class ATOTestConfig():
    def __init__(self, input_path, order_max, product_max, IO_max, BO_max, demand_type, T=200):
        input_path = input_path

        # problem setting parameters
        self.is_inf = True

        self.T = T
        self.action_order_min = 0  # minimum order quantity for each components
        self.action_order_max = order_max  # maximum order quantity for each components
        self.action_product_min = 0  # minimum order quantity for each components
        self.action_product_max = product_max  # maximum order quantity for each components

        self.c_h = pd.read_excel(input_path, sheet_name='ch', header=None).to_numpy().reshape(
            -1)  # holding cost for each component
        self.c_p = pd.read_excel(input_path, sheet_name='cp', header=None).to_numpy().reshape(
            -1)  # backorder cost for each product
        self.L = pd.read_excel(input_path, sheet_name='L', header=None).to_numpy().reshape(-1)  # leadtime of components
        self.NoProducts = self.c_p.size  # total number of products
        self.NoComponents = self.c_h.size  # total number of components
        self.point = [0] + [sum(self.L[:i + 1]) for i in range(self.NoComponents)]
        self.demand_dist = pd.read_excel(input_path, sheet_name='demand', header=None).to_numpy().reshape(
            -1)
        if demand_type == 2:
            self.demand_dist = pd.read_excel(input_path, sheet_name='demand', header=None).to_numpy()
        self.demand_type = demand_type # demand distribution, 0 for uniform dist, 1 for poisson dist
        self.IT_min = self.action_order_min
        self.IT_max = self.action_order_max
        self.IO_min = 0  # minimum IO for each components
        self.IO_max = IO_max  # maximum IO for each components
        self.BO_min = 0  # minimum BO for each components
        self.BO_max = BO_max  # maximum BO for each components
        self.action_low = np.array(
            [self.action_order_min for i in range(self.NoComponents)] + [self.action_product_min for j in
                                                                         range(self.NoProducts)])
        self.action_high = np.array(
            [self.action_order_max for i in range(self.NoComponents)] + [self.action_product_max for j in
                                                                         range(self.NoProducts)])
        self.observation_low = np.array(
            [self.IT_min for i in range(sum(self.L))] + [self.IO_min for i in range(self.NoComponents)] + [self.BO_min
                                                                                                           for i in
                                                                                                           range(
                                                                                                               self.NoProducts)])
        self.observation_high = np.array(
            [self.IT_max for i in range(sum(self.L))] + [self.IO_max for i in range(self.NoComponents)] + [self.BO_max
                                                                                                           for i in
                                                                                                           range(
                                                                                                               self.NoProducts)])
        self.usage_rate = np.asmatrix(pd.read_excel(input_path, sheet_name='BOM', header=None).to_numpy(),
                                 dtype=int)  # usage rate matrix for components and products
        self.index = [sum(self.L[:i + 1]) - 1 for i in range(self.NoComponents)]

class ATOTrainConfig():
    def __init__(self, input_path, order_max=100, product_max=100, IO_max=100, BO_max=100, T=200, demand_type=0, demand_set = None):
        input_path = input_path

        # problem setting parameters
        self.is_inf = True
        self.demand_set = demand_set

        self.T = T
        self.action_order_min = 0  # minimum order quantity for each components
        self.action_order_max = order_max  # maximum order quantity for each components
        self.action_product_min = 0  # minimum order quantity for each components
        self.action_product_max = product_max  # maximum order quantity for each components

        self.c_h = pd.read_excel(input_path, sheet_name='ch', header=None).to_numpy().reshape(
            -1)  # holding cost for each component
        self.c_p = pd.read_excel(input_path, sheet_name='cp', header=None).to_numpy().reshape(
            -1)  # backorder cost for each product
        self.L = pd.read_excel(input_path, sheet_name='L', header=None).to_numpy().reshape(-1)  # leadtime of components
        self.NoProducts = self.c_p.size  # total number of products
        self.NoComponents = self.c_h.size  # total number of components
        self.point = [0] + [sum(self.L[:i + 1]) for i in range(self.NoComponents)]
        self.demand_dist = pd.read_excel(input_path, sheet_name='demand', header=None).to_numpy().reshape(
            -1)
        self.demand_type = demand_type # demand distribution, 0 for uniform dist, 1 for poisson dist
        self.IT_min = self.action_order_min
        self.IT_max = self.action_order_max
        self.IO_min = 0  # minimum IO for each components
        self.IO_max = IO_max  # maximum IO for each components
        self.BO_min = 0  # minimum BO for each components
        self.BO_max = BO_max  # maximum BO for each components
        self.action_low = np.array(
            [self.action_order_min for i in range(self.NoComponents)] + [self.action_product_min for j in
                                                                         range(self.NoProducts)])
        self.action_high = np.array(
            [self.action_order_max for i in range(self.NoComponents)] + [self.action_product_max for j in
                                                                         range(self.NoProducts)])
        self.observation_low = np.array(
            [self.IT_min for i in range(sum(self.L))] + [self.IO_min for i in range(self.NoComponents)] + [self.BO_min
                                                                                                           for i in
                                                                                                           range(
                                                                                                               self.NoProducts)])
        self.observation_high = np.array(
            [self.IT_max for i in range(sum(self.L))] + [self.IO_max for i in range(self.NoComponents)] + [self.BO_max
                                                                                                           for i in
                                                                                                           range(
                                                                                                               self.NoProducts)])
        self.usage_rate = np.asmatrix(pd.read_excel(input_path, sheet_name='BOM', header=None).to_numpy(),
                                 dtype=int)  # usage rate matrix for components and products
        self.index = [sum(self.L[:i + 1]) - 1 for i in range(self.NoComponents)]
