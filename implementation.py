import numpy as np
import pulp

class DivisionPropertyOptimizer:
    def __init__(self, M):
        self.M_np = np.array(M)
        self.N = len(M)
        self.c = self.M_np.sum(axis=0)
        self.r = self.M_np.sum(axis=1)
        self.prob = pulp.LpProblem("DivisionProperty", pulp.LpMinimize)
        self.x = pulp.LpVariable.dicts("x", range(self.N), cat='Binary')
        self.y = pulp.LpVariable.dicts("y", range(self.N), cat='Binary')
        self.t = pulp.LpVariable.dicts("t", range(self.c.sum()), cat='Binary')
        self._build_model()
        
    def I(self, i, j):
        column = self.M_np[i].nonzero()[0][j]
        at_column = sum(self.M_np.T[column].nonzero()[0] < i)
        before = sum(self.c[k] for k in range(column))
        return before + at_column
    
    def _build_model(self):
        print("Total number of slack variables: ", len(self.t))
        
        # Objective function (not really used here, to be set later)
        self.prob += 0
        
        # Constraints
        for i in range(self.N):
            start = 0 if i == 0 else start + self.c[i - 1]
            self.prob += self.x[i] == pulp.lpSum([self.t[start + j] for j in range(self.c[i])])
        for i in range(self.N):
            self.prob += self.y[i] == pulp.lpSum([self.t[self.I(i, j)] for j in range(self.r[i])])

    def solve(self):
        self.prob.solve()
        return {
            'status': pulp.LpStatus[self.prob.status],
            'x': [pulp.value(self.x[i]) for i in range(self.N)],
            'y': [pulp.value(self.y[i]) for i in range(self.N)],
            't': [pulp.value(self.t[i]) for i in range(self.c.sum())]
        }