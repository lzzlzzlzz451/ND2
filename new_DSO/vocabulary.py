import numpy as np
from ND2.utils import AttrDict
 
class Vocabulary:
    def __init__(self, config):
        self.config = config
        voc = config.vocabulary
 
        # 手动构建 word2id（AttrDict 不支持 + 合并）
        self.word2id = {}
        for section in [voc.special, voc.placeholder,
                        voc.constant, voc.variable.node, voc.variable.edge,
                        voc.operator.binary, voc.operator.unary]:
            for k, v in section.items():
                self.word2id[k] = v
        # 补充系数 token
        self.word2id['<C>'] = voc.coefficient
        self.word2id['<Cv>'] = voc.node_coefficient
        self.word2id['<Ce>'] = voc.edge_coefficient
 
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.n_words = len(self.word2id)
        self.pad_id = voc.special.pad
        self.sos_id = voc.special.sos
        self.eos_id = voc.special.eos
 
        # ---- 以下部分不能漏 ----
        self._arity = {}
        self._kind = {}
        self._type_scope = {}
        for token, tid in self.word2id.items():
            if token in voc.special:
                self._arity[tid] = 0; self._kind[tid] = 'special'; self._type_scope[tid] = 'any'
            elif token in voc.placeholder:
                self._arity[tid] = 0; self._kind[tid] = 'placeholder'; self._type_scope[tid] = token
            elif token == '<C>':
                self._arity[tid] = 0; self._kind[tid] = 'coefficient'; self._type_scope[tid] = 'scalar'
            elif token == '<Cv>':
                self._arity[tid] = 0; self._kind[tid] = 'node_coeff'; self._type_scope[tid] = 'node'
            elif token == '<Ce>':
                self._arity[tid] = 0; self._kind[tid] = 'edge_coeff'; self._type_scope[tid] = 'edge'
            elif token in voc.variable.node:
                self._arity[tid] = 0; self._kind[tid] = 'variable'; self._type_scope[tid] = 'node'
            elif token in voc.variable.edge:
                self._arity[tid] = 0; self._kind[tid] = 'variable'; self._type_scope[tid] = 'edge'
            elif token in voc.constant:
                self._arity[tid] = 0; self._kind[tid] = 'constant'; self._type_scope[tid] = 'scalar'
            elif token in voc.operator.binary:
                self._arity[tid] = 2; self._kind[tid] = 'operator'; self._type_scope[tid] = 'any'
            elif token in voc.operator.unary:
                self._arity[tid] = 1; self._kind[tid] = 'operator'
                self._type_scope[tid] = {'aggr': 'node', 'sour': 'edge', 'targ': 'edge'}.get(token, 'any')
 
        max_id = max(self.word2id.values())
        self.arity_array = np.zeros(max_id + 1, dtype=np.int32)
        for tid, a in self._arity.items():
            self.arity_array[tid] = a
 
    def arity(self, tid): return self._arity.get(tid, 0)
    def kind(self, tid): return self._kind.get(tid, 'unknown')
    def type_scope(self, tid): return self._type_scope.get(tid, 'any')
    def tokens_of_kind(self, kind): return [tid for tid, k in self._kind.items() if k == kind]
    def is_terminal(self, tid): return self._arity[tid] == 0 and self._kind[tid] != 'placeholder'