import torch
import torch.nn as nn
import numpy as np
import pdb

class StackRNN(object):
    def __init__(self, cell, initial_state, if_cell=True):
        """
        if_cell: support LSTM that function like LSTMcell
        """
        if if_cell:
            self.real_cell = cell
        else:
            self.fake_cell = cell
            self.real_cell = self.cell
        self.initial_state=initial_state
        self.state = [initial_state]

    def push(self, vec):
        """
        state <hidden_vec, cell_vec>
        """
        vec = vec.unsqueeze(0)
        self.state.append(self.real_cell(vec, self.state[-1]))

    def pop(self):
        top_state = self.state.pop()
        return top_state[0].squeeze(0)

    def embedding(self):
        assert len(self.state) > 0
        return self.state[-1][0].squeeze(0)

    @DeprecationWarning
    def cell(self, input_v, state):
        input_v = input_v.unsqueeze(0)
        state = (state[0].unsqueeze(0), state[1].unsqueeze(0))
        ouput, new_state = self.fake_cell(input_v, state)
        return (new_state[0].squeeze(0), new_state[1].squeeze(0))

    def __len__(self):
        return len(self.state) - 1



class BufferR(object):
    def __init__(self, vecs,  empty):
        """
        vecs with dimention sent_len * dim, input is already reversed
        embeds with dimention sent_len * input_dim, input is already reversed
        """
        #print (vecs.size())
        sent_len, dim = vecs.size()
        self.buffer = vecs
        self.sent_len = sent_len
        self.empty = empty
        self.pointer = sent_len - 1
    
    def embedding(self):
        """
        funciton like top
        """
        if self.pointer < 0:
            return self.empty
        else:
            return self.buffer[self.pointer]
        
    def pop(self):
        """
        pop the buffer lstm and also the embeds
        """
        if self.pointer < 0:
            raise ValueError
        popped_buffer = self.buffer[self.pointer]

        self.pointer -= 1
        return popped_buffer
    def back(self,index):
        '''
        back and buffer pointer+1
        '''
        self.pointer+=index-1

    def __len__(self):
        return self.pointer + 1

