import abc

class BaseModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, HP):
        self.HP = HP

        #Abstract instance variables that have to be defined by the network
        self.learning_rate = None
        self.train = None
        # self.compute_cost = None
        self.predict = None
        self.ouput = None
        self.reset_states = None
        self.net = None

        self.get_probs = None
        self.get_probs_flat = None

        # self.save_model = None
        # self.load_model = None

        self.create_network()

    @abc.abstractmethod
    def create_network(self):
        '''
        Create networks.
        Needs to define the abstact instance variables
        '''
        return None
