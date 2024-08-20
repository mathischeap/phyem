# -*- coding: utf-8 -*-
"""
We use this class to store method that generate tables from data of instances of type
`Matrix3dInputRunner` or `ThreeInputsRunner`.

<unittest> <unittests_P_Solvers> <test_No4_M3IR>.

Yi Zhang (C)
Created on Wed May  8 17:46:17 2019
Aerodynamics, AE
TU Delft
"""
import numpy as np
from tools.frozen import Frozen
from tools.decorators.all import accepts

class M_TIR_Tabulate(Frozen):
    """ 
    <unittest> <unittests_P_Solvers> <test_No4_M3IR>.
    
    """
    @accepts('self', ('Matrix3dInputRunner', 'ThreeInputsRunner'))
    def __init__(self, M3irTir):
        """ """
        self._tir_ = M3irTir
        self._freeze()

    @accepts('self', str)
    def array(self, opn):
        """ 
        The most standard tabular method. We make a dictionary of 2d arrays (tables) 
        for the output named `opn`.
        
        Special cases:
            If self._tir_ is a `ThreeInputsRunner`:
                Keys are indices along I3seq.
            If self._tir_ is a `Matrix3dInputRunner`:
                Keys are the indices along the third axes of the input_matrix.
        
        <unittest> <unittests_P_Solvers> <test_No4_M3IR>.
        
        Parameters
        ----------
        opn :
            
        Returns
        -------
        d : dict
            Keys are number of variables of the third input. Like if we run a runner 
            for (p, K, c) and p = [....], K = [...], c=[0, 0.3], then in the d, keys
            are 0 and 1, and they refer to c=0 and c=0.3.
            
        """
        assert opn in self._tir_.output_names, \
            " <Tabular> : {} is not in output_names: {}.".format(opn, self._tir_.output_names)
        d = {}
        df = self._tir_.rdf
        #___ Matrix3dInputRunner ______________________________________________________
        if self._tir_.__class__.__name__ == 'Matrix3dInputRunner':
            if self._tir_._input_shape_ is None:
                I0seq = list(set(df[self._tir_.input_names[0]]))
                I0seq.sort()
                isp0 = len(I0seq)
                I1seq = list(set(df[self._tir_.input_names[1]]))
                I1seq.sort()
                isp1 = len(I1seq)
                I2seq = list(set(df[self._tir_.input_names[2]]))
                I2seq.sort()
                isp2 = len(I2seq)
                isp = (isp0, isp1, isp2)
                _I0Seq_ = initialize_3d_list(*isp)
                _I1Seq_ = initialize_3d_list(*isp)
                _I2Seq_ = initialize_3d_list(*isp)
                for i in range(isp0):
                    for j in range(isp1):
                        for k in range(isp2):
                            _I0Seq_[i][j][k] = I0seq[i]
                            _I1Seq_[i][j][k] = I1seq[j]
                            _I2Seq_[i][j][k] = I2seq[k]
            else:
                isp = self._tir_._input_shape_
                _I0Seq_ = self._tir_._I0Seq_
                _I1Seq_ = self._tir_._I1Seq_
                _I2Seq_ = self._tir_._I2Seq_
        #___ ThreeInputsRunner ________________________________________________________
        elif self._tir_.__class__.__name__ == 'ThreeInputsRunner':
                isp0 = len(self._tir_.I0seq)
                isp1 = len(self._tir_.I1seq)
                isp2 = len(self._tir_.I2seq)
                isp = (isp0, isp1, isp2)
                _I0Seq_ = self._tir_.I0seq
                _I1Seq_ = self._tir_.I1seq
                _I2Seq_ = self._tir_.I2seq
        #______ ELSE: ERRORING ________________________________________________________
        else:
            raise Exception(" <Tabular> : this Tabular not for it.")
        #------------------------------------------------------------------------------
        for k in range(isp[2]): # go throug all keys
            d[k] = np.zeros(isp[:2]) # distribute memory 
        ipn = self._tir_.input_names
        #___ Matrix3dInputRunner ______________________________________________________
        if self._tir_.__class__.__name__ == 'Matrix3dInputRunner':
            for k in range(isp[2]): # go throug all keys
                for i in range(isp[0]):
                    for j in range(isp[1]):
                        dft = df[df[ipn[0]]==_I0Seq_[i][j][k]]
                        dft = dft[dft[ipn[1]]==_I1Seq_[i][j][k]]
                        dft = dft[dft[ipn[2]]==_I2Seq_[i][j][k]]
                        if dft.empty:
                            d[k][i][j] = np.NaN
                        else:
                            d[k][i][j] = dft.iloc[0][opn]
        #___ ThreeInputsRunner ________________________________________________________
        elif self._tir_.__class__.__name__ == 'ThreeInputsRunner':
            for k in range(isp[2]): # go throug all keys
                dfk = df[df[ipn[2]]==_I2Seq_[k]]
                for i in range(isp[0]):
                    dfi = dfk[dfk[ipn[0]]==_I0Seq_[i]]
                    for j in range(isp[1]):
                        dfj = dfi[dfi[ipn[1]]==_I1Seq_[j]]
                        if dfj.empty:
                            d[k][i][j] = np.NaN
                        else:
                            d[k][i][j] = dfj.iloc[0][opn]
        #------------------------------------------------------------------------------
        else:
            raise Exception(" <Tabular> : this Tabular not for it.")
        return d

def initialize_3d_list(a, b, c):
    """

    :param a:
    :param b:
    :param c:
    :return:
    """
    lst = [[[None for _ in range(c)] for _ in range(b)] for _ in range(a)]
    return lst
