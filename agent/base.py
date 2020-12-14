class Agent:
    def __init__(self):
        pass
    def select_move(self, mat):
        raise NotImplementedError

class Encoder:
    def __init__(self):
        pass
    def name(self):
        raise NotImplementedError
    def encode(self):
        raise NotImplementedError
    def encode_point(self):
        raise NotImplementedError
    def num_points(self):
        raise NotImplementedError
    def shape(self):
        raise NotImplementedError