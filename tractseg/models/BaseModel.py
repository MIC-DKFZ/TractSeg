# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import six

#Python 2 and 3 compatible
@six.add_metaclass(abc.ABCMeta)
class BaseModel():
# class BaseModel(metaclass=abc.ABCMeta):

    def __init__(self, HP):
        self.HP = HP

        #Abstract instance variables that have to be defined by the network
        self.train = None
        self.predict = None
        self.net = None
        self.get_probs = None
        self.save_model = None
        self.load_model = None

        self.create_network()

    @abc.abstractmethod
    def create_network(self):
        '''
        Create networks.
        Needs to define the abstact instance variables
        '''
        return None
