# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2023 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`AbrahamsonEtAl2015`
               :class:`AbrahamsonEtAl2015SInter`
               :class:`AbrahamsonEtAl2015SInterHigh`
               :class:`AbrahamsonEtAl2015SInterLow`
               :class:`AbrahamsonEtAl2015SSlab`
               :class:`AbrahamsonEtAl2015SSlabHigh`
               :class:`AbrahamsonEtAl2015SSlabLow`

Added the basin scaling term from CB14.

"""
# import numpy as np

# from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
# from openquake.hazardlib import const
# from openquake.hazardlib.imt import PGA, SA



import numpy as np
import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


# Make a class for the CoeffsTable
class CoeffsTable(object):
    """ Class to mimicking the behavior of the openquake CoeffsTable. The class
    implements the call method so that the following works:
        
        C = CoeffsTable(coeff_table_string) # initialize the object
        C(T = 0) #returns the dictionary of coefficients for PGA (T = 0)
        C(1.0)['c1i'] = 8.10 # returns coefficient c1i for Sa(T = 1.0s)
        
        Note: openQuake uses call C[1.0]['c1i'], i.e., there the CoeffsTable
        object acts like a dictionary of dictionaries. This is the difference 
        here where, e.g., C(1.0)['c1i'] would be used. 
        
        
    If the specified period is not amongst the periods for which the GMPE is 
    defined, then the coefficients get linearly interpolated in the log of the 
    periods.
    
    Parameters
    ----------
    coeff_table_string : string of coefficients as used in openquake, e.g., for
        Abrahamson and Gulerce 2020 the following string is used:
            
     coeff_table_string = ""\
     imt     c1i   vlin       b        a1       a2       a6      a7      a8    a10     a11      a12      a13      a14     a16    a17      a18    a19      a20     a21     a22     a23     a24      a25      a26      a27     a28      a29     a30      a31      a32      a33      a34      a35      a36      a37    a39     a41  USA-AK_Adj  CAS_Adj     d1     d2   rhoW   rhoB  phi_s2s_g1  phi_s2s_g2   phi_s2s_g3      e1       e2      e3
     pga    8.20   865.1  -1.186   4.5960  -1.4500  -0.0043  3.2100  0.0440  3.210  0.0070   0.9000   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0400  0.0400  0.0000  0.0015   0.0007   0.0036  -0.0004  0.0025   0.0006  0.0033   3.7783   3.3468   3.8025   5.0361   4.6272   4.8044   3.5669  0.000  -0.029       0.487    0.828  0.325  0.137  1.000  1.000       0.396       0.396        0.545   0.550   -0.270   0.050
     0.010  8.20   865.1  -1.186   4.5960  -1.4500  -0.0043  3.2100  0.0440  3.210  0.0070   0.9000   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0400  0.0400  0.0000  0.0015   0.0007   0.0036  -0.0004  0.0025   0.0006  0.0033   3.7783   3.3468   3.8025   5.0361   4.6272   4.8044   3.5669  0.000  -0.029       0.487    0.828  0.325  0.137  1.000  1.000       0.396       0.396        0.545   0.550   -0.270   0.050
     0.020  8.20   865.1  -1.219   4.6780  -1.4500  -0.0043  3.2100  0.0440  3.210  0.0070   1.0080   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0400  0.0400  0.0000  0.0015   0.0006   0.0036  -0.0005  0.0025   0.0005  0.0033   3.8281   3.4401   3.9053   5.1375   4.6958   4.8943   3.6425  0.000  -0.024       0.519    0.825  0.325  0.137  0.990  0.990       0.396       0.396        0.545   0.550   -0.270   0.050
     0.030  8.20   907.8  -1.273   4.7730  -1.4500  -0.0044  3.2100  0.0440  3.210  0.0070   1.1270   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0400  0.0400  0.0000  0.0015   0.0006   0.0037  -0.0007  0.0025   0.0005  0.0034   3.8933   3.5087   4.0189   5.2699   4.7809   5.0028   3.7063  0.000  -0.034       0.543    0.834  0.325  0.137  0.990  0.990       0.396       0.396        0.545   0.550   -0.270   0.050
     0.050  8.20  1053.5  -1.346   5.0290  -1.4500  -0.0046  3.2100  0.0440  3.210  0.0070   1.3330   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0400  0.0400  0.0000  0.0011   0.0006   0.0039  -0.0009  0.0026   0.0004  0.0036   4.2867   3.6553   4.2952   5.6157   5.0211   5.2819   3.9184  0.000  -0.061       0.435    0.895  0.325  0.137  0.970  0.985       0.396       0.467        0.644   0.560   -0.270   0.050
     0.075  8.20  1085.7  -1.471   5.3340  -1.4500  -0.0047  3.2100  0.0440  3.210  0.0070   1.5650   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.0600  0.0600  0.0000  0.0011   0.0004   0.0039  -0.0009  0.0026   0.0003  0.0037   4.5940   3.9799   4.5464   6.0204   5.3474   5.6123   4.2207  0.000  -0.076       0.410    0.863  0.325  0.137  0.950  0.980       0.396       0.516        0.713   0.580   -0.270   0.050
     0.100  8.20  1032.5  -1.624   5.4550  -1.4500  -0.0048  3.2100  0.0440  3.210  0.0070   1.6790   0.0000  -0.4600  0.0900  0.000  -0.2000  0.000   0.0000  0.1000  0.1000  0.0000  0.0012   0.0003   0.0039  -0.0008  0.0026   0.0003  0.0038   4.7077   4.1312   4.6138   6.1625   5.5065   5.7668   4.3536  0.000  -0.049       0.397    0.842  0.325  0.137  0.920  0.970       0.396       0.516        0.713   0.590   -0.270   0.050
     0.150  8.20   877.6  -1.931   5.3760  -1.4250  -0.0047  3.2100  0.0440  3.210  0.0070   1.8530   0.0000  -0.4600  0.0900  0.000  -0.1860  0.000  -0.0550  0.1350  0.1350  0.0690  0.0013  -0.0002   0.0037  -0.0009  0.0022   0.0001  0.0037   4.6065   4.2737   4.5290   5.9614   5.5180   5.7313   4.3664  0.000  -0.026       0.428    0.737  0.325  0.137  0.900  0.960       0.396       0.516        0.647   0.590   -0.270   0.050
     0.200  8.20   748.2  -2.188   4.9360  -1.3350  -0.0045  3.2100  0.0430  3.210  0.0062   2.0220   0.0000  -0.4600  0.0840  0.000  -0.1500  0.000  -0.1050  0.1700  0.1700  0.1400  0.0013  -0.0007   0.0031  -0.0010  0.0018  -0.0001  0.0035   4.1866   3.9650   4.1656   5.3920   5.1668   5.2943   4.0169  0.000  -0.011       0.442    0.746  0.325  0.137  0.870  0.940       0.396       0.516        0.596   0.570   -0.270   0.050
     0.250  8.20   654.3  -2.381   4.6360  -1.2750  -0.0043  3.2100  0.0420  3.210  0.0056   2.1810   0.0000  -0.4600  0.0800  0.000  -0.1400  0.000  -0.1340  0.1700  0.1700  0.1640  0.0013  -0.0009   0.0027  -0.0011  0.0016  -0.0003  0.0033   3.8515   3.6821   3.9147   5.0117   4.8744   5.0058   3.7590  0.101  -0.009       0.494    0.796  0.325  0.137  0.840  0.930       0.396       0.501        0.539   0.530   -0.224   0.043
     0.300  8.20   587.1  -2.518   4.4230  -1.2310  -0.0042  3.2100  0.0410  3.210  0.0051   2.2810  -0.0020  -0.4600  0.0780  0.000  -0.1200  0.000  -0.1500  0.1700  0.1700  0.1900  0.0014  -0.0010   0.0020  -0.0009  0.0014  -0.0002  0.0032   3.5783   3.5415   3.7846   4.7057   4.6544   4.7588   3.5914  0.184   0.005       0.565    0.782  0.325  0.137  0.820  0.910       0.396       0.488        0.488   0.490   -0.186   0.037
     0.400  8.20   503.0  -2.657   4.1240  -1.1650  -0.0040  3.2100  0.0400  3.210  0.0043   2.3790  -0.0070  -0.4700  0.0750  0.000  -0.1000  0.000  -0.1500  0.1700  0.1700  0.2060  0.0015  -0.0010   0.0013  -0.0007  0.0011   0.0000  0.0030   3.2493   3.3256   3.5702   4.2896   4.3660   4.3789   3.3704  0.315   0.040       0.625    0.768  0.325  0.137  0.740  0.860       0.396       0.468        0.468   0.425   -0.126   0.028
     0.500  8.20   456.6  -2.669   3.8380  -1.1150  -0.0037  3.2100  0.0390  3.210  0.0037   2.3390  -0.0110  -0.4800  0.0720  0.000  -0.0800  0.000  -0.1500  0.1700  0.1700  0.2200  0.0015  -0.0011   0.0009  -0.0007  0.0008   0.0002  0.0027   2.9818   3.1334   3.3552   3.9322   4.0779   4.0394   3.1564  0.416   0.097       0.634    0.728  0.325  0.137  0.660  0.800       0.396       0.451        0.451   0.375   -0.079   0.022
     0.600  8.20   430.3  -2.599   3.5620  -1.0710  -0.0035  3.2100  0.0380  3.210  0.0033   2.2170  -0.0150  -0.4900  0.0700  0.000  -0.0600  0.000  -0.1500  0.1700  0.1700  0.2250  0.0015  -0.0012   0.0006  -0.0007  0.0006   0.0002  0.0025   2.7784   2.9215   3.0922   3.6149   3.8146   3.7366   2.9584  0.499   0.145       0.581    0.701  0.325  0.137  0.590  0.780       0.396       0.438        0.438   0.345   -0.041   0.016
     0.750  8.15   410.5  -2.401   3.1520  -1.0200  -0.0032  3.2100  0.0370  3.210  0.0027   1.9460  -0.0210  -0.5000  0.0670  0.000  -0.0470  0.000  -0.1500  0.1700  0.1700  0.2170  0.0014  -0.0011   0.0003  -0.0007  0.0004   0.0002  0.0022   2.4780   2.5380   2.6572   3.1785   3.4391   3.2930   2.6556  0.600   0.197       0.497    0.685  0.325  0.137  0.500  0.730       0.396       0.420        0.420   0.300    0.005   0.009
     1.000  8.10   400.0  -1.955   2.5440  -0.9500  -0.0029  3.2100  0.0350  3.210  0.0019   1.4160  -0.0280  -0.5100  0.0630  0.000  -0.0350  0.000  -0.1500  0.1700  0.1700  0.1850  0.0013  -0.0008   0.0001  -0.0008  0.0002   0.0001  0.0019   1.9252   1.9626   2.1459   2.5722   2.8056   2.6475   2.0667  0.731   0.269       0.469    0.642  0.325  0.137  0.410  0.690       0.396       0.396        0.396   0.240    0.065   0.000
     1.500  8.05   400.0  -1.025   1.6360  -0.8600  -0.0026  3.2100  0.0340  3.210  0.0008   0.3940  -0.0410  -0.5200  0.0590  0.000  -0.0180  0.000  -0.1300  0.1700  0.1700  0.0830  0.0014  -0.0004  -0.0001  -0.0008  0.0001   0.0000  0.0016   0.9924   1.3568   1.3499   1.6499   1.8546   1.6842   1.3316  0.748   0.347       0.509    0.325  0.312  0.113  0.330  0.620       0.379       0.379        0.379   0.230    0.065   0.000
     2.000  8.00   400.0  -0.299   1.0760  -0.8200  -0.0024  3.2100  0.0320  3.210  0.0000  -0.4170  -0.0500  -0.5300  0.0590  0.000  -0.0100  0.000  -0.1100  0.1700  0.1700  0.0450  0.0015   0.0002   0.0000  -0.0007  0.0002   0.0000  0.0014   0.4676   0.8180   0.8148   1.0658   1.3020   1.1002   0.7607  0.761   0.384       0.478    0.257  0.302  0.096  0.300  0.560       0.366       0.366        0.366   0.230    0.065   0.000
     2.500  7.95   400.0   0.000   0.6580  -0.7980  -0.0022  3.2100  0.0310  3.210  0.0000  -0.7250  -0.0570  -0.5400  0.0600  0.000  -0.0050  0.000  -0.0950  0.1700  0.1700  0.0260  0.0014   0.0004   0.0000  -0.0007  0.0002  -0.0002  0.0012   0.0579   0.4389   0.3979   0.6310   0.8017   0.6737   0.3648  0.770   0.397       0.492    0.211  0.295  0.082  0.270  0.520       0.356       0.356        0.356   0.230    0.065   0.000
     3.000  7.90   400.0   0.000   0.4240  -0.7930  -0.0021  3.1300  0.0300  3.130  0.0000  -0.6950  -0.0650  -0.5400  0.0590  0.000   0.0000  0.000  -0.0850  0.1700  0.1700  0.0350  0.0014   0.0007   0.0003  -0.0007  0.0004  -0.0002  0.0011  -0.1391   0.1046   0.1046   0.3882   0.5958   0.4126   0.1688  0.778   0.404       0.470    0.296  0.289  0.072  0.250  0.495       0.348       0.348        0.348   0.240    0.065   0.000
     4.000  7.85   400.0   0.000   0.0930  -0.7930  -0.0020  2.9850  0.0290  2.985  0.0000  -0.6380  -0.0770  -0.5400  0.0500  0.000   0.0000  0.000  -0.0730  0.1700  0.1700  0.0530  0.0014   0.0010   0.0007  -0.0006  0.0006  -0.0002  0.0010  -0.3030  -0.1597  -0.2324   0.0164   0.3522   0.0097  -0.0323  0.790   0.397       0.336    0.232  0.280  0.055  0.220  0.430       0.335       0.335        0.335   0.270    0.065   0.000
     5.000  7.80   400.0   0.000  -0.1450  -0.7930  -0.0020  2.8180  0.0280  2.818  0.0000  -0.5970  -0.0880  -0.5400  0.0430  0.000   0.0000  0.000  -0.0650  0.1700  0.1700  0.0720  0.0014   0.0013   0.0014  -0.0004  0.0008  -0.0001  0.0010  -0.4094  -0.2063  -0.5722  -0.2802   0.1874  -0.2715  -0.1516  0.799   0.378       0.228    0.034  0.273  0.041  0.190  0.400       0.324       0.324        0.324   0.300    0.065   0.000
     6.000  7.80   400.0   0.000  -0.3200  -0.7930  -0.0020  2.6820  0.0270  2.682  0.0000  -0.5610  -0.0980  -0.5400  0.0380  0.000   0.0000  0.000  -0.0600  0.1700  0.1700  0.0860  0.0014   0.0015   0.0015  -0.0003  0.0011   0.0000  0.0010  -0.5010  -0.3223  -0.8631  -0.4822  -0.1243  -0.4591  -0.2217  0.807   0.358       0.151   -0.037  0.267  0.030  0.170  0.370       0.314       0.314        0.314   0.320    0.065   0.000
     7.500  7.80   400.0   0.000  -0.5560  -0.7930  -0.0020  2.5150  0.0260  2.515  0.0000  -0.5300  -0.1100  -0.5400  0.0320  0.000   0.0000  0.000  -0.0550  0.1700  0.1700  0.1150  0.0014   0.0017   0.0015  -0.0002  0.0014   0.0001  0.0010  -0.6209  -0.4223  -1.1773  -0.7566  -0.3316  -0.6822  -0.3338  0.817   0.333       0.051   -0.178  0.259  0.017  0.140  0.320       0.301       0.301        0.301   0.350    0.065   0.000
     10.00  7.80   400.0   0.000  -0.8600  -0.7930  -0.0020  2.3000  0.0250  2.300  0.0000  -0.4860  -0.1270  -0.5400  0.0240  0.000   0.0000  0.000  -0.0450  0.1700  0.1700  0.1510  0.0014   0.0017   0.0015  -0.0001  0.0017   0.0002  0.0010  -0.6221  -0.5909  -1.4070  -1.0870  -0.6783  -0.9173  -0.5441  0.829   0.281      -0.251   -0.313  0.250  0.000  0.100  0.280       0.286       0.286        0.286   0.350    0.065   0.000
     ""       
    
    
    """
    def __init__(self, coeff_table_string):
        coeff_table_string_oq = StringIO(coeff_table_string)
        self.df = pd.read_table(coeff_table_string_oq, delim_whitespace = True, 
                                index_col=0)
        self.periods = self.get_periods()
        self.df.index = self.periods #change index values form str to floats, 
        # 'pga' is set to 0
        
    def __call__(self, T):
        # todo : implement interpolation if period is non-existant
        if( T in self.periods):
            return self.df.loc[T].to_dict()
        elif(T > min(self.periods) and T < max(self.periods)):
            # interpolation
            return self.get_coefficients(T)
        else:
            raise ValueError('Period outside of period range bounds.')

    def get_periods(self):
        """Extract periods from indices, convert to floats"""
        per_str_lst = self.df.index.to_list()
        per_lst = [0.0 if x=='pga' else float(x) for x in per_str_lst]
        return per_lst

    def get_coefficients(self, T):
        """Obtain coefficients by linear interpolation in the log of periods."""
        # Find closest period beneath and above T
        per_low = max( [ x if x < T else -999 for x in self.periods ]  )
        per_high = min( [ x if x > T else 999 for x in self.periods ]  )
        # to avoid division by zero if using T = 0 (PGA)
        if(per_low < 0.01):
            return self(0.01)
            # Note: coefficients for PGA (T = 0) and T = 0.01s are the same for
            #       this GMPE, so I return coefficients for T = 0.01s; works 
            #       here, but in general this is not correct
        # linear interpolation in log of the period
        coeffs_low = self(per_low)
        coeffs_high = self(per_high)
        ln_ratio = np.log(per_high/per_low)
        delta_ln_ratio = np.log(T/per_low)
        coeffs_interp = {}
        for key in coeffs_low.keys():
            # print(key)
            coeffs_interp[key] = coeffs_low[key] + delta_ln_ratio*(coeffs_high[key] - coeffs_low[key])/ln_ratio
        return coeffs_interp



# Period-Independent Coefficients (Table 2)
CONSTS = {
    'n': 1.18,
    'c': 1.88,
    'theta3': 0.1,
    'theta4': 0.9,
    'theta5': 0.0,
    'theta9': 0.4,
    'c4': 10.0,
    'C1': 7.8
}

C1 = 7.2  # for Montalva2017

# Total epistemic uncertainty factors from Abrahamson et al. (2018)

table_string ="""\
    imt     SIGMA_MU_SINTER    SIGMA_MU_SSLAB
    pga                 0.3              0.50
    0.010               0.3              0.50
    0.020               0.3              0.50
    0.030               0.3              0.50
    0.050               0.3              0.50
    0.075               0.3              0.50
    0.100               0.3              0.50
    0.150               0.3              0.50
    0.200               0.3              0.50
    0.250               0.3              0.46
    0.300               0.3              0.42
    0.400               0.3              0.38
    0.500               0.3              0.34
    0.600               0.3              0.30
    0.750               0.3              0.30
    1.000               0.3              0.30
    1.500               0.3              0.30
    2.000               0.3              0.30
    2.500               0.3              0.30
    3.000               0.3              0.30
    4.000               0.3              0.30
    5.000               0.3              0.30
    6.000               0.3              0.30
    7.500               0.3              0.30
    10.00               0.3              0.30
    """
BCHYDRO_SIGMA_MU = CoeffsTable(table_string)

def get_stress_factor(imt, slab):
    """
    Returns the stress adjustment factor for the BC Hydro GMPE according to
    Abrahamson et al. (2018)
    """
    C = BCHYDRO_SIGMA_MU(imt)
    return (C["SIGMA_MU_SSLAB"] if slab else C["SIGMA_MU_SINTER"]) / 1.65


def _compute_magterm(C1, theta1, theta4, theta5, theta13, dc1, mag):
    """
    Computes the magnitude scaling term given by equation (2)
    corrected by a local adjustment factor
    """
    base = theta1 + theta4 * dc1
    dmag = C1 + dc1
    f_mag = np.where(
        mag > dmag, theta5 * (mag - dmag), theta4 * (mag - dmag))
    return base + f_mag + theta13 * (10. - mag) ** 2.


# theta6_adj used in BCHydro
def _compute_disterm(trt, C1, theta2, theta14, theta3, ctx, c4, theta9,
                     theta6_adj, theta6, theta10):
    if trt == 'subduction_interface':#const.TRT.SUBDUCTION_INTERFACE:
        dists = ctx.rrup
        assert theta10 == 0., theta10
    elif trt == 'subduction_inslab':#const.TRT.SUBDUCTION_INTRASLAB:
        dists = ctx.rhypo
    else:
        raise NotImplementedError(trt)
    return (theta2 + theta14 + theta3 * (ctx.mag - C1)) * np.log(
        dists + c4 * np.exp((ctx.mag - 6.) * theta9)) + (
        theta6_adj + theta6) * dists + theta10


def _compute_forearc_backarc_term(trt, faba_model, C, ctx):
    if trt == 'subduction_interface':#const.TRT.SUBDUCTION_INTERFACE:
        dists = ctx.rrup
        a, b = C['theta15'], C['theta16']
        min_dist = 100.
    elif trt == 'subduction_inslab':#const.TRT.SUBDUCTION_INTRASLAB:
        dists = ctx.rhypo
        a, b = C['theta7'], C['theta8']
        min_dist = 85.
    else:
        raise NotImplementedError(trt)
    if faba_model is None:
        backarc = np.bool_(ctx.backarc)
        f_faba = np.zeros_like(dists)
        # Term only applies to backarc ctx (F_FABA = 0. for forearc)
        fixed_dists = dists[backarc]
        fixed_dists[fixed_dists < min_dist] = min_dist
        f_faba[backarc] = a + b * np.log(fixed_dists / 40.)
        return f_faba

    # in BCHydro subclasses
    fixed_dists = np.copy(dists)
    fixed_dists[fixed_dists < min_dist] = min_dist
    f_faba = a + b * np.log(fixed_dists / 40.)
    return f_faba * faba_model(-ctx.xvf)


def _compute_distance_term(kind, trt, theta6_adj, C, ctx):
    """
    Computes the distance scaling term, as contained within equation (1)
    """
    if kind.startswith("montalva"):
        theta3 = C['theta3']
    else:
        theta3 = CONSTS['theta3']
    if kind == "montalva17":
        C1 = 7.2
    else:
        C1 = 7.8
    if trt == 'subduction_interface':#const.TRT.SUBDUCTION_INTERFACE:
        return _compute_disterm(
            trt, C1, C['theta2'], 0., theta3, ctx, CONSTS['c4'],
            CONSTS['theta9'], theta6_adj, C['theta6'], theta10=0.)
    else:  # sslab
        return _compute_disterm(
            trt, C1, C['theta2'], C['theta14'], theta3, ctx,
            CONSTS['c4'], CONSTS['theta9'], theta6_adj, C['theta6'],
            C["theta10"])


def _compute_focal_depth_term(trt, C, ctx):
    """
    Computes the hypocentral depth scaling term - as indicated by
    equation (3)
    For interface events F_EVENT = 0.. so no depth scaling is returned.
    For SSlab events computes the hypocentral depth scaling term as
    indicated by equation (3)
    """
    if trt == 'subduction_interface':#const.TRT.SUBDUCTION_INTERFACE:
        return np.zeros_like(ctx.mag)
    z_h = np.clip(ctx.hypo_depth, None, 120.)
    return C['theta11'] * (z_h - 60.)


def _compute_magnitude_term(kind, C, dc1, mag):
    """
    Computes the magnitude scaling term given by equation (2)
    """
    if kind == "base":
        return _compute_magterm(
            CONSTS['C1'], C['theta1'], CONSTS['theta4'],
            CONSTS['theta5'], C['theta13'], dc1, mag)
    elif kind == "montalva16":
        return _compute_magterm(
            CONSTS['C1'], C['theta1'], C['theta4'],
            C['theta5'], C['theta13'], dc1, mag)
    elif kind == "montalva17":
        return _compute_magterm(C1, C['theta1'], C['theta4'],
                                C['theta5'], 0., dc1, mag)


def _compute_pga_rock(kind, trt, theta6_adj, faba_model, C, dc1, ctx):
    """
    Compute and return mean imt value for rock conditions
    (vs30 = 1000 m/s)
    """
    mean = (_compute_magnitude_term(kind, C, dc1, ctx.mag) +
            _compute_distance_term(kind, trt, theta6_adj, C, ctx) +
            _compute_focal_depth_term(trt, C, ctx) +
            _compute_forearc_backarc_term(trt, faba_model, C, ctx))
    # Apply linear site term
    site_response = ((C['theta12'] + C['b'] * CONSTS['n']) *
                     np.log(1000. / C['vlin']))
    return mean + site_response


def _compute_site_response_term(C, ctx, pga1000):
    """
    Compute and return site response model term
    This GMPE adopts the same site response scaling model of
    Walling et al (2008) as implemented in the Abrahamson & Silva (2008)
    GMPE. The functional form is retained here.
    """
    vs_star = ctx.vs30.copy()
    vs_star[vs_star > 1000.0] = 1000.
    arg = vs_star / C["vlin"]
    site_resp_term = C["theta12"] * np.log(arg)
    # Get linear scaling term
    idx = ctx.vs30 >= C["vlin"]
    site_resp_term[idx] += (C["b"] * CONSTS["n"] * np.log(arg[idx]))
    # Get nonlinear scaling term
    idx = np.logical_not(idx)
    site_resp_term[idx] += (
        -C["b"] * np.log(pga1000[idx] + CONSTS["c"]) +
        C["b"] * np.log(pga1000[idx] + CONSTS["c"] *
                        (arg[idx] ** CONSTS["n"])))
    return site_resp_term

# basin scaling term from CB14
def _select_basin_model(SJ, vs30):
    """
    Select the preferred basin model (California or Japan) to scale
    basin depth with respect to Vs30
    """
    if SJ:
        # Japan Basin Model - Equation 34 of Campbell & Bozorgnia (2014)
        return np.exp(5.359 - 1.102 * np.log(vs30))
    else:
        # California Basin Model - Equation 33 of
        # Campbell & Bozorgnia (2014)
        return np.exp(7.089 - 1.144 * np.log(vs30))


def _get_basin_response_term(SJ, C, z2pt5):
    """
    Returns the basin response term defined in equation 20
    """
    f_sed = np.zeros(len(z2pt5))
    idx = z2pt5 < 1.0
    f_sed[idx] = (C["c14"] + C["c15"] * SJ) * (z2pt5[idx] - 1.0)
    idx = z2pt5 > 3.0
    f_sed[idx] = C["c16"] * C["k3"] * np.exp(-0.75) * (
        1. - np.exp(-0.25 * (z2pt5[idx] - 3.)))
    return f_sed


def scale_factor_basin_usgs(T, T_low = 0.5, T_high = 1.0):
    """ Scaling factor to multiply the basin scaling term when implementing
        the USGS approach. Basin depth scaling only applied for periods above
        1.0s; below 0.5s no basin scaling; between 0.5s and 0.75s linear 
        interploation. """
    return np.min( [(T_high - T) / T_low , 1.0] )


class AbrahamsonEtAl2015SInter(object):
    """
    Implements the Subduction GMPE developed by Norman Abrahamson, Nicholas
    Gregor and Kofi Addo, otherwise known as the "BC Hydro" Model, published
    as "BC Hydro Ground Motion Prediction Equations For Subduction Earthquakes
    (2015, Earthquake Spectra, in press), for subduction interface events.

    From observations of very large events it was found that the magnitude
    scaling term can be adjusted as part of the epistemic uncertainty model.
    The adjustment comes in the form of the parameter DeltaC1, which is
    period dependent for interface events. To capture the epistemic uncertainty
    in DeltaC1, three models are proposed: a 'central', 'upper' and 'lower'
    model. The current class implements the 'central' model, whilst additional
    classes will implement the 'upper' and 'lower' alternatives.
    """
    # #: Supported tectonic region type is subduction interface
    # DEFINED_FOR_TECTONIC_REGION_TYPE = trt = const.TRT.SUBDUCTION_INTERFACE

    # #: Supported intensity measure types are spectral acceleration,
    # #: and peak ground acceleration
    # DEFINED_FOR_INTENSITY_MEASURE_TYPES = {PGA, SA}

    # #: Supported intensity measure component is the geometric mean component
    # DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN

    # #: Supported standard deviation types are inter-event, intra-event
    # #: and total, see table 3, pages 12 - 13
    # DEFINED_FOR_STANDARD_DEVIATION_TYPES = {
    #     const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT}

    # #: Site amplification is dependent upon Vs30
    # #: For the Abrahamson et al (2013) GMPE a new term is introduced to
    # #: determine whether a site is on the forearc with respect to the
    # #: subduction interface, or on the backarc. This boolean is a vector
    # #: containing True for a backarc site or False for a forearc or
    # #: unknown site.
    # REQUIRES_SITES_PARAMETERS = {'vs30', 'backarc', 'z2pt5'}

    # #: Required rupture parameters are magnitude for the interface model
    # REQUIRES_RUPTURE_PARAMETERS = {'mag'}

    # #: Required distance measure is closest distance to rupture, for
    # #: interface events
    # REQUIRES_DISTANCES = {'rrup'}

    #: Reference soil conditions (bottom of page 29)
    DEFINED_FOR_REFERENCE_VELOCITY = 1000

    delta_c1 = None
    kind = "base"
    FABA_ALL_MODELS = {}  # overridden in BCHydro

    def __init__(self, T, Rup, Site, is_scale_to_rotd100 = True, **kwargs):
        super().__init__(**kwargs)
        # self.ergodic = kwargs.get('ergodic', True)
        # self.theta6_adj = kwargs.get("theta6_adjustment", 0.0)
        # self.sigma_mu_epsilon = kwargs.get("sigma_mu_epsilon", 0.0)
        self.ergodic = True
        self.theta6_adj = 0.0
        self.sigma_mu_epsilon = 0.0
                
        # faba_type = kwargs.get("faba_taper_model", "Step")
        # if 'xvf' in self.REQUIRES_SITES_PARAMETERS:  # BCHydro subclasses
        #     self.faba_model = self.FABA_ALL_MODELS[faba_type](**kwargs)
        # else:
        #     self.faba_model = None
        self.trt = Rup.trt
        self.faba_model = None
        self.is_scale_to_rotd100 = is_scale_to_rotd100
        self.period1 = T
        self.median, self.sigma = self.get_median_and_sigma(T, Rup, Site)

    def fn_rotd50_2_rotd100(self, T):
        """ Function to compute scaling factor which converts RotD50 to RotD100 
        following the approach of Boore and Kishida (2017), specifically using
        equation (2) and coefficients from Table 2. 
        
        Reference
            Boore and Kishida (2017) Relations between some horizontal-component 
            ground-motion intensity measures used in practice    
        
        Parameters
        ----------
        T: float, period for which to compute the scaling factor
        
        Returns
        -------
        float
        
        """
        
        # # Regression coefficients for RotD100/RotD50
        R_i = [0, 1.188, 1.225, 1.241, 1.287, 1.287]
        T_i = [0, 0.12,  0.41,  3.14,  10.0,  0  ]
        
        # Regression coefficients for Larger/RotD50
        # R_i = [0, 1.107, 1.133, 1.149, 1.178, 1.178]
        # T_i = [0, 0.10,  0.45,  4.36,  8.78,  0  ]
            
        # Compute the scaling factors    
        a = R_i[1] + ( ( R_i[2] - R_i[1] ) / np.log(T_i[2]/T_i[1]) )*np.log(T / T_i[1]) 
        b = R_i[2] + ( ( R_i[3] - R_i[2] ) / np.log(T_i[3]/T_i[2]) )*np.log(T / T_i[2]) 
        c = np.min([ a, b ])
        d_1 = R_i[3] + ( ( R_i[4] - R_i[3] ) / np.log(T_i[4]/T_i[3]) )*np.log(T / T_i[3])
        d = np.min([ d_1, R_i[5] ])
        e = np.max([ c, d ])
        ratio = np.max( [R_i[1], e] )
          
        return ratio        
        
    def get_median_and_sigma(self, T, Rup, Site): #compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        
        
        trt = Rup.trt #self.DEFINED_FOR_TECTONIC_REGION_TYPE
        C_PGA = self.COEFFS(T=0) #self.COEFFS[PGA()]
        
        # # define ctx and other parameters as:
        # :param ctx: a RuptureContext object or a numpy recarray of size N
        # :param imts: a list of M Intensity Measure Types
        # :param mean: an array of shape (M, N) for the means
        # :param sig: an array of shape (M, N) for the TOTAL stddevs
        # :param tau: an array of shape (M, N) for the INTER_EVENT stddevs
        # :param phi: an array of shape (M, N) for the INTRA_EVENT stddevs
        
        # define ctx as an empty recarray, then set values like so:
        # x = np.recarray((1,), dtype=[('ztor', float), ('rrup', float), ('mag', float)]) 
        # x.rrup = 5.5
        ctx = np.recarray((1,), dtype=[('ztor', float), ('rrup', float), 
                                       ('vs30', float), ('mag', float),
                                       ('z2pt5', float), ('backarc', bool) ]) 
        ctx.ztor = Rup.Ztor
        ctx.rrup = Rup.Rrup
        ctx.vs30 = Site.Vs30
        ctx.mag = Rup.M
        ctx.z2pt5 = Site.Z25
        ctx.backarc = Site.backarc
        
        
        # create imts from T --- just make it a list of periods; define other arrays as (M, 1)
        # at the end return median and sigma -- median is np.exp of the mean
        imts = [ x for x in T ] # list of M periods
        mean = np.empty(shape=( len(imts), 1))
        sig = np.empty(shape=( len(imts), 1))
        tau = np.empty(shape=( len(imts), 1))
        phi = np.empty(shape=( len(imts), 1))

        # computation happens here
        
        dc1_pga = self.delta_c1 or self.COEFFS_MAG_SCALE(T=0)["dc1"] # dc1_pga = self.delta_c1 or self.COEFFS_MAG_SCALE[PGA()]["dc1"]
        # compute median pga on rock (vs30=1000), needed for site response
        # term calculation
        pga1000 = np.exp(_compute_pga_rock(
            self.kind, self.trt, self.theta6_adj, self.faba_model,
            C_PGA, dc1_pga, ctx))
        for m, imt in enumerate(imts):
            C = self.COEFFS(imt)
            dc1 = self.delta_c1 or self.COEFFS_MAG_SCALE(imt)["dc1"]
            mean[m] = (
                _compute_magnitude_term(
                    self.kind, C, dc1, ctx.mag) +
                _compute_distance_term(
                    self.kind, self.trt, self.theta6_adj, C, ctx) +
                _compute_focal_depth_term(
                    self.trt, C, ctx) +
                _compute_forearc_backarc_term(
                    self.trt, self.faba_model, C, ctx) +
                _compute_site_response_term(
                    C, ctx, pga1000) + 
                #add the basin scaling term here
                _get_basin_response_term(SJ = 0, 
                        C = self.COEFFS_basin(imt), z2pt5 = ctx.z2pt5))
            
            # remove basin term depending on the period
            if( imt < 1.0 ):
                sf_temp = scale_factor_basin_usgs(imt)
                mean[m] -= sf_temp * _get_basin_response_term(SJ = 0, 
                        C = self.COEFFS_basin(imt), z2pt5 = ctx.z2pt5)
            
            # conversion to RotD100
            if(self.is_scale_to_rotd100):
                mean[m] = np.log( np.exp(mean[m]) * self.fn_rotd50_2_rotd100(imt) )
            
            
            if self.sigma_mu_epsilon:
                sigma_mu = get_stress_factor(
                    imt, trt == 'subduction_inslab')
                mean[m] += sigma_mu * self.sigma_mu_epsilon

            sig[m] = C["sigma"] if self.ergodic else C["sigma_ss"]
            tau[m] = C['tau']
            phi[m] = C["phi"] if self.ergodic else np.sqrt(
                C["sigma_ss"] ** 2. - C["tau"] ** 2.)
        
        # modification so that the output works with the rest of the scripts
        mean = np.array([x[0] for x in mean])
        sig = np.array([x[0] for x in sig])
        # print(sig)
        # print(mean)
        median = np.exp(mean)        
        sigma = sig
        
        return median, sigma


    # Period-dependent coefficients (Table 3)
    table_string ="""\
    imt          vlin        b   theta1    theta2    theta6   theta7    theta8  theta10  theta11   theta12   theta13   theta14  theta15   theta16      phi     tau   sigma  sigma_ss
    pga      865.1000  -1.1860   4.2203   -1.3500   -0.0012   1.0988   -1.4200   3.1200   0.0130    0.9800   -0.0135   -0.4000   0.9969   -1.0000   0.6000  0.4300  0.7400    0.6000
    0.0200   865.1000  -1.1860   4.2203   -1.3500   -0.0012   1.0988   -1.4200   3.1200   0.0130    0.9800   -0.0135   -0.4000   0.9969   -1.0000   0.6000  0.4300  0.7400    0.6000
    0.0500  1053.5000  -1.3460   4.5371   -1.4000   -0.0012   1.2536   -1.6500   3.3700   0.0130    1.2880   -0.0138   -0.4000   1.1030   -1.1800   0.6000  0.4300  0.7400    0.6000
    0.0750  1085.7000  -1.4710   5.0733   -1.4500   -0.0012   1.4175   -1.8000   3.3700   0.0130    1.4830   -0.0142   -0.4000   1.2732   -1.3600   0.6000  0.4300  0.7400    0.6000
    0.1000  1032.5000  -1.6240   5.2892   -1.4500   -0.0012   1.3997   -1.8000   3.3300   0.0130    1.6130   -0.0145   -0.4000   1.3042   -1.3600   0.6000  0.4300  0.7400    0.6000
    0.1500   877.6000  -1.9310   5.4563   -1.4500   -0.0014   1.3582   -1.6900   3.2500   0.0130    1.8820   -0.0153   -0.4000   1.2600   -1.3000   0.6000  0.4300  0.7400    0.6000
    0.2000   748.2000  -2.1880   5.2684   -1.4000   -0.0018   1.1648   -1.4900   3.0300   0.0129    2.0760   -0.0162   -0.3500   1.2230   -1.2500   0.6000  0.4300  0.7400    0.6000
    0.2500   654.3000  -2.3810   5.0594   -1.3500   -0.0023   0.9940   -1.3000   2.8000   0.0129    2.2480   -0.0172   -0.3100   1.1600   -1.1700   0.6000  0.4300  0.7400    0.6000
    0.3000   587.1000  -2.5180   4.7945   -1.2800   -0.0027   0.8821   -1.1800   2.5900   0.0128    2.3480   -0.0183   -0.2800   1.0500   -1.0600   0.6000  0.4300  0.7400    0.6000
    0.4000   503.0000  -2.6570   4.4644   -1.1800   -0.0035   0.7046   -0.9800   2.2000   0.0127    2.4270   -0.0206   -0.2300   0.8000   -0.7800   0.6000  0.4300  0.7400    0.6000
    0.5000   456.6000  -2.6690   4.0181   -1.0800   -0.0044   0.5799   -0.8200   1.9200   0.0125    2.3990   -0.0231   -0.1900   0.6620   -0.6200   0.6000  0.4300  0.7400    0.6000
    0.6000   430.3000  -2.5990   3.6055   -0.9900   -0.0050   0.5021   -0.7000   1.7000   0.0124    2.2730   -0.0256   -0.1600   0.5800   -0.5000   0.6000  0.4300  0.7400    0.6000
    0.7500   410.5000  -2.4010   3.2174   -0.9100   -0.0058   0.3687   -0.5400   1.4200   0.0120    1.9930   -0.0296   -0.1200   0.4800   -0.3400   0.6000  0.4300  0.7400    0.6000
    1.0000   400.0000  -1.9550   2.7981   -0.8500   -0.0062   0.1746   -0.3400   1.1000   0.0114    1.4700   -0.0363   -0.0700   0.3300   -0.1400   0.6000  0.4300  0.7400    0.6000
    1.5000   400.0000  -1.0250   2.0123   -0.7700   -0.0064  -0.0820   -0.0500   0.7000   0.0100    0.4080   -0.0493    0.0000   0.3100    0.0000   0.6000  0.4300  0.7400    0.6000
    2.0000   400.0000  -0.2990   1.4128   -0.7100   -0.0064  -0.2821    0.1200   0.7000   0.0085   -0.4010   -0.0610    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    2.5000   400.0000   0.0000   0.9976   -0.6700   -0.0064  -0.4108    0.2500   0.7000   0.0069   -0.7230   -0.0711    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    3.0000   400.0000   0.0000   0.6443   -0.6400   -0.0064  -0.4466    0.3000   0.7000   0.0054   -0.6730   -0.0798    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    4.0000   400.0000   0.0000   0.0657   -0.5800   -0.0064  -0.4344    0.3000   0.7000   0.0027   -0.6270   -0.0935    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    5.0000   400.0000   0.0000  -0.4624   -0.5400   -0.0064  -0.4368    0.3000   0.7000   0.0005   -0.5960   -0.0980    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    6.0000   400.0000   0.0000  -0.9809   -0.5000   -0.0064  -0.4586    0.3000   0.7000  -0.0013   -0.5660   -0.0980    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    7.5000   400.0000   0.0000  -1.6017   -0.4600   -0.0064  -0.4433    0.3000   0.7000  -0.0033   -0.5280   -0.0980    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    10.0000  400.0000   0.0000  -2.2937   -0.4000   -0.0064  -0.4828    0.3000   0.7000  -0.0060   -0.5040   -0.0980    0.0000   0.3000    0.0000   0.6000  0.4300  0.7400    0.6000
    """
    COEFFS = CoeffsTable(table_string)

    table_string="""\
    IMT    dc1
    pga    0.2
    0.02   0.2
    0.30   0.2
    0.50   0.1
    1.00   0.0
    2.00  -0.1
    3.00  -0.2
    10.0  -0.2
    """
    COEFFS_MAG_SCALE = CoeffsTable(table_string)
    
    # these are the coefficients that should be used for the basin term from CB14
    # table_string="""\
    #     IMT         c0      c1       c2       c3       c4       c5      c6      c7       c9     c10      c11      c12     c13       c14      c15     c16       c17      c18       c19       c20     Dc20      a2      h1      h2       h3       h5       h6     k1       k2      k3    phi1    phi2    tau1    tau2    phiC   rholny
    #     pgv     -2.895   1.510    0.270   -1.299   -0.453   -2.466   0.204   5.837   -0.168   0.305    1.713    2.602   2.457    0.1060    0.332   0.585    0.0517   0.0327   0.00613   -0.0017   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -1.955   1.929   0.655   0.494   0.317   0.297   0.190    0.684
    #     pga     -4.416   0.984    0.537   -1.499   -0.496   -2.773   0.248   6.768   -0.212   0.720    1.090    2.186   1.420   -0.0064   -0.202   0.393    0.0977   0.0333   0.00757   -0.0055   0.0000   0.167   0.241   1.474   -0.715   -0.337   -0.270    865   -1.186   1.839   0.734   0.492   0.409   0.322   0.166    1.000
    #     0.01    -4.365   0.977    0.533   -1.485   -0.499   -2.773   0.248   6.753   -0.214   0.720    1.094    2.191   1.416   -0.0070   -0.207   0.390    0.0981   0.0334   0.00755   -0.0055   0.0000   0.168   0.242   1.471   -0.714   -0.336   -0.270    865   -1.186   1.839   0.734   0.492   0.404   0.325   0.166    1.000
    #     0.02    -4.348   0.976    0.549   -1.488   -0.501   -2.772   0.247   6.502   -0.208   0.730    1.149    2.189   1.453   -0.0167   -0.199   0.387    0.1009   0.0327   0.00759   -0.0055   0.0000   0.166   0.244   1.467   -0.711   -0.339   -0.263    865   -1.219   1.840   0.738   0.496   0.417   0.326   0.166    0.998
    #     0.03    -4.024   0.931    0.628   -1.494   -0.517   -2.782   0.246   6.291   -0.213   0.759    1.290    2.164   1.476   -0.0422   -0.202   0.378    0.1095   0.0331   0.00790   -0.0057   0.0000   0.167   0.246   1.467   -0.713   -0.338   -0.259    908   -1.273   1.841   0.747   0.503   0.446   0.344   0.165    0.986
    #     0.05    -3.479   0.887    0.674   -1.388   -0.615   -2.791   0.240   6.317   -0.244   0.826    1.449    2.138   1.549   -0.0663   -0.339   0.295    0.1226   0.0270   0.00803   -0.0063   0.0000   0.173   0.251   1.449   -0.701   -0.338   -0.263   1054   -1.346   1.843   0.777   0.520   0.508   0.377   0.162    0.938
    #     0.075   -3.293   0.902    0.726   -1.469   -0.596   -2.745   0.227   6.861   -0.266   0.815    1.535    2.446   1.772   -0.0794   -0.404   0.322    0.1165   0.0288   0.00811   -0.0070   0.0000   0.198   0.260   1.435   -0.695   -0.347   -0.219   1086   -1.471   1.845   0.782   0.535   0.504   0.418   0.158    0.887
    #     0.10    -3.666   0.993    0.698   -1.572   -0.536   -2.633   0.210   7.294   -0.229   0.831    1.615    2.969   1.916   -0.0294   -0.416   0.384    0.0998   0.0325   0.00744   -0.0073   0.0000   0.174   0.259   1.449   -0.708   -0.391   -0.201   1032   -1.624   1.847   0.769   0.543   0.445   0.426   0.170    0.870
    #     0.15    -4.866   1.267    0.510   -1.669   -0.490   -2.458   0.183   8.031   -0.211   0.749    1.877    3.544   2.161    0.0642   -0.407   0.417    0.0760   0.0388   0.00716   -0.0069   0.0000   0.198   0.254   1.461   -0.715   -0.449   -0.099    878   -1.931   1.852   0.769   0.543   0.382   0.387   0.180    0.876
    #     0.20    -5.411   1.366    0.447   -1.750   -0.451   -2.421   0.182   8.385   -0.163   0.764    2.069    3.707   2.465    0.0968   -0.311   0.404    0.0571   0.0437   0.00688   -0.0060   0.0000   0.204   0.237   1.484   -0.721   -0.393   -0.198    748   -2.188   1.856   0.761   0.552   0.339   0.338   0.186    0.870
    #     0.25    -5.962   1.458    0.274   -1.711   -0.404   -2.392   0.189   7.534   -0.150   0.716    2.205    3.343   2.766    0.1441   -0.172   0.466    0.0437   0.0463   0.00556   -0.0055   0.0000   0.185   0.206   1.581   -0.787   -0.339   -0.210    654   -2.381   1.861   0.744   0.545   0.340   0.316   0.191    0.850
    #     0.30    -6.403   1.528    0.193   -1.770   -0.321   -2.376   0.195   6.990   -0.131   0.737    2.306    3.334   3.011    0.1597   -0.084   0.528    0.0323   0.0508   0.00458   -0.0049   0.0000   0.164   0.210   1.586   -0.795   -0.447   -0.121    587   -2.518   1.865   0.727   0.568   0.340   0.300   0.198    0.819
    #     0.40    -7.566   1.739   -0.020   -1.594   -0.426   -2.303   0.185   7.012   -0.159   0.738    2.398    3.544   3.203    0.1410    0.085   0.540    0.0209   0.0432   0.00401   -0.0037   0.0000   0.160   0.226   1.544   -0.770   -0.525   -0.086    503   -2.657   1.874   0.690   0.593   0.356   0.264   0.206    0.743
    #     0.50    -8.379   1.872   -0.121   -1.577   -0.440   -2.296   0.186   6.902   -0.153   0.718    2.355    3.016   3.333    0.1474    0.233   0.638    0.0092   0.0405   0.00388   -0.0027   0.0000   0.184   0.217   1.554   -0.770   -0.407   -0.281    457   -2.669   1.883   0.663   0.611   0.379   0.263   0.208    0.684
    #     0.75    -9.841   2.021   -0.042   -1.757   -0.443   -2.232   0.186   5.522   -0.090   0.795    1.995    2.616   3.054    0.1764    0.411   0.776   -0.0082   0.0420   0.00420   -0.0016   0.0000   0.216   0.154   1.626   -0.780   -0.371   -0.285    410   -2.401   1.906   0.606   0.633   0.430   0.326   0.221    0.562
    #     1.00   -11.011   2.180   -0.069   -1.707   -0.527   -2.158   0.169   5.650   -0.105   0.556    1.447    2.470   2.562    0.2593    0.479   0.771   -0.0131   0.0426   0.00409   -0.0006   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -1.955   1.929   0.579   0.628   0.470   0.353   0.225    0.467
    #     1.50   -12.469   2.270    0.047   -1.621   -0.630   -2.063   0.158   5.795   -0.058   0.480    0.330    2.108   1.453    0.2881    0.566   0.748   -0.0187   0.0380   0.00424    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -1.025   1.974   0.541   0.603   0.497   0.399   0.222    0.364
    #     2.00   -12.969   2.271    0.149   -1.512   -0.768   -2.104   0.158   6.632   -0.028   0.401   -0.514    1.327   0.657    0.3112    0.562   0.763   -0.0258   0.0252   0.00448    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -0.299   2.019   0.529   0.588   0.499   0.400   0.226    0.298
    #     3.00   -13.306   2.150    0.368   -1.315   -0.890   -2.051   0.148   6.759    0.000   0.206   -0.848    0.601   0.367    0.3478    0.534   0.686   -0.0311   0.0236   0.00345    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.110   0.527   0.578   0.500   0.417   0.229    0.234
    #     4.00   -14.020   2.132    0.726   -1.506   -0.885   -1.986   0.135   7.978    0.000   0.105   -0.793    0.568   0.306    0.3747    0.522   0.691   -0.0413   0.0102   0.00603    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.200   0.521   0.559   0.543   0.393   0.237    0.202
    #     5.00   -14.558   2.116    1.027   -1.721   -0.878   -2.021   0.135   8.538    0.000   0.000   -0.748    0.356   0.268    0.3382    0.477   0.670   -0.0281   0.0034   0.00805    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.291   0.502   0.551   0.534   0.421   0.237    0.184
    #     7.50   -15.509   2.223    0.169   -0.756   -1.077   -2.179   0.165   8.468    0.000   0.000   -0.664    0.075   0.374    0.3754    0.321   0.757   -0.0205   0.0050   0.00280    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.517   0.457   0.546   0.523   0.438   0.271    0.176
    #     10.0   -15.975   2.132    0.367   -0.800   -1.282   -2.244   0.180   6.564    0.000   0.000   -0.576   -0.027   0.297    0.3506    0.174   0.621    0.0009   0.0099   0.00458    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.744   0.441   0.543   0.466   0.438   0.290    0.154
    #     """
    table_string="""\
        IMT         c0      c1       c2       c3       c4       c5      c6      c7       c9     c10      c11      c12     c13       c14      c15     c16       c17      c18       c19       c20     Dc20      a2      h1      h2       h3       h5       h6     k1       k2      k3    phi1    phi2    tau1    tau2    phiC   rholny
        pga     -4.416   0.984    0.537   -1.499   -0.496   -2.773   0.248   6.768   -0.212   0.720    1.090    2.186   1.420   -0.0064   -0.202   0.393    0.0977   0.0333   0.00757   -0.0055   0.0000   0.167   0.241   1.474   -0.715   -0.337   -0.270    865   -1.186   1.839   0.734   0.492   0.409   0.322   0.166    1.000
        0.01    -4.365   0.977    0.533   -1.485   -0.499   -2.773   0.248   6.753   -0.214   0.720    1.094    2.191   1.416   -0.0070   -0.207   0.390    0.0981   0.0334   0.00755   -0.0055   0.0000   0.168   0.242   1.471   -0.714   -0.336   -0.270    865   -1.186   1.839   0.734   0.492   0.404   0.325   0.166    1.000
        0.02    -4.348   0.976    0.549   -1.488   -0.501   -2.772   0.247   6.502   -0.208   0.730    1.149    2.189   1.453   -0.0167   -0.199   0.387    0.1009   0.0327   0.00759   -0.0055   0.0000   0.166   0.244   1.467   -0.711   -0.339   -0.263    865   -1.219   1.840   0.738   0.496   0.417   0.326   0.166    0.998
        0.03    -4.024   0.931    0.628   -1.494   -0.517   -2.782   0.246   6.291   -0.213   0.759    1.290    2.164   1.476   -0.0422   -0.202   0.378    0.1095   0.0331   0.00790   -0.0057   0.0000   0.167   0.246   1.467   -0.713   -0.338   -0.259    908   -1.273   1.841   0.747   0.503   0.446   0.344   0.165    0.986
        0.05    -3.479   0.887    0.674   -1.388   -0.615   -2.791   0.240   6.317   -0.244   0.826    1.449    2.138   1.549   -0.0663   -0.339   0.295    0.1226   0.0270   0.00803   -0.0063   0.0000   0.173   0.251   1.449   -0.701   -0.338   -0.263   1054   -1.346   1.843   0.777   0.520   0.508   0.377   0.162    0.938
        0.075   -3.293   0.902    0.726   -1.469   -0.596   -2.745   0.227   6.861   -0.266   0.815    1.535    2.446   1.772   -0.0794   -0.404   0.322    0.1165   0.0288   0.00811   -0.0070   0.0000   0.198   0.260   1.435   -0.695   -0.347   -0.219   1086   -1.471   1.845   0.782   0.535   0.504   0.418   0.158    0.887
        0.10    -3.666   0.993    0.698   -1.572   -0.536   -2.633   0.210   7.294   -0.229   0.831    1.615    2.969   1.916   -0.0294   -0.416   0.384    0.0998   0.0325   0.00744   -0.0073   0.0000   0.174   0.259   1.449   -0.708   -0.391   -0.201   1032   -1.624   1.847   0.769   0.543   0.445   0.426   0.170    0.870
        0.15    -4.866   1.267    0.510   -1.669   -0.490   -2.458   0.183   8.031   -0.211   0.749    1.877    3.544   2.161    0.0642   -0.407   0.417    0.0760   0.0388   0.00716   -0.0069   0.0000   0.198   0.254   1.461   -0.715   -0.449   -0.099    878   -1.931   1.852   0.769   0.543   0.382   0.387   0.180    0.876
        0.20    -5.411   1.366    0.447   -1.750   -0.451   -2.421   0.182   8.385   -0.163   0.764    2.069    3.707   2.465    0.0968   -0.311   0.404    0.0571   0.0437   0.00688   -0.0060   0.0000   0.204   0.237   1.484   -0.721   -0.393   -0.198    748   -2.188   1.856   0.761   0.552   0.339   0.338   0.186    0.870
        0.25    -5.962   1.458    0.274   -1.711   -0.404   -2.392   0.189   7.534   -0.150   0.716    2.205    3.343   2.766    0.1441   -0.172   0.466    0.0437   0.0463   0.00556   -0.0055   0.0000   0.185   0.206   1.581   -0.787   -0.339   -0.210    654   -2.381   1.861   0.744   0.545   0.340   0.316   0.191    0.850
        0.30    -6.403   1.528    0.193   -1.770   -0.321   -2.376   0.195   6.990   -0.131   0.737    2.306    3.334   3.011    0.1597   -0.084   0.528    0.0323   0.0508   0.00458   -0.0049   0.0000   0.164   0.210   1.586   -0.795   -0.447   -0.121    587   -2.518   1.865   0.727   0.568   0.340   0.300   0.198    0.819
        0.40    -7.566   1.739   -0.020   -1.594   -0.426   -2.303   0.185   7.012   -0.159   0.738    2.398    3.544   3.203    0.1410    0.085   0.540    0.0209   0.0432   0.00401   -0.0037   0.0000   0.160   0.226   1.544   -0.770   -0.525   -0.086    503   -2.657   1.874   0.690   0.593   0.356   0.264   0.206    0.743
        0.50    -8.379   1.872   -0.121   -1.577   -0.440   -2.296   0.186   6.902   -0.153   0.718    2.355    3.016   3.333    0.1474    0.233   0.638    0.0092   0.0405   0.00388   -0.0027   0.0000   0.184   0.217   1.554   -0.770   -0.407   -0.281    457   -2.669   1.883   0.663   0.611   0.379   0.263   0.208    0.684
        0.75    -9.841   2.021   -0.042   -1.757   -0.443   -2.232   0.186   5.522   -0.090   0.795    1.995    2.616   3.054    0.1764    0.411   0.776   -0.0082   0.0420   0.00420   -0.0016   0.0000   0.216   0.154   1.626   -0.780   -0.371   -0.285    410   -2.401   1.906   0.606   0.633   0.430   0.326   0.221    0.562
        1.00   -11.011   2.180   -0.069   -1.707   -0.527   -2.158   0.169   5.650   -0.105   0.556    1.447    2.470   2.562    0.2593    0.479   0.771   -0.0131   0.0426   0.00409   -0.0006   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -1.955   1.929   0.579   0.628   0.470   0.353   0.225    0.467
        1.50   -12.469   2.270    0.047   -1.621   -0.630   -2.063   0.158   5.795   -0.058   0.480    0.330    2.108   1.453    0.2881    0.566   0.748   -0.0187   0.0380   0.00424    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -1.025   1.974   0.541   0.603   0.497   0.399   0.222    0.364
        2.00   -12.969   2.271    0.149   -1.512   -0.768   -2.104   0.158   6.632   -0.028   0.401   -0.514    1.327   0.657    0.3112    0.562   0.763   -0.0258   0.0252   0.00448    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400   -0.299   2.019   0.529   0.588   0.499   0.400   0.226    0.298
        3.00   -13.306   2.150    0.368   -1.315   -0.890   -2.051   0.148   6.759    0.000   0.206   -0.848    0.601   0.367    0.3478    0.534   0.686   -0.0311   0.0236   0.00345    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.110   0.527   0.578   0.500   0.417   0.229    0.234
        4.00   -14.020   2.132    0.726   -1.506   -0.885   -1.986   0.135   7.978    0.000   0.105   -0.793    0.568   0.306    0.3747    0.522   0.691   -0.0413   0.0102   0.00603    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.200   0.521   0.559   0.543   0.393   0.237    0.202
        5.00   -14.558   2.116    1.027   -1.721   -0.878   -2.021   0.135   8.538    0.000   0.000   -0.748    0.356   0.268    0.3382    0.477   0.670   -0.0281   0.0034   0.00805    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.291   0.502   0.551   0.534   0.421   0.237    0.184
        7.50   -15.509   2.223    0.169   -0.756   -1.077   -2.179   0.165   8.468    0.000   0.000   -0.664    0.075   0.374    0.3754    0.321   0.757   -0.0205   0.0050   0.00280    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.517   0.457   0.546   0.523   0.438   0.271    0.176
        10.0   -15.975   2.132    0.367   -0.800   -1.282   -2.244   0.180   6.564    0.000   0.000   -0.576   -0.027   0.297    0.3506    0.174   0.621    0.0009   0.0099   0.00458    0.0000   0.0000   0.596   0.117   1.616   -0.733   -0.128   -0.756    400    0.000   2.744   0.441   0.543   0.466   0.438   0.290    0.154
        """
    COEFFS_basin = CoeffsTable(table_string)


class AbrahamsonEtAl2015SSlab(AbrahamsonEtAl2015SInter):
    """
    Implements the Subduction GMPE developed by Norman Abrahamson, Nicholas
    Gregor and Kofi Addo, otherwise known as the "BC Hydro" Model, published
    as "BC Hydro Ground Motion Prediction Equations For Subduction Earthquakes
    (2013, Earthquake Spectra, in press).
    This implements only the inslab GMPE. For inslab events the source is
    considered to be a point source located at the hypocentre. Therefore
    the hypocentral distance metric is used in place of the rupture distance,
    and the hypocentral depth is used to scale the ground motion by depth
    """
    # #: Supported tectonic region type is subduction in-slab
    # DEFINED_FOR_TECTONIC_REGION_TYPE = trt = const.TRT.SUBDUCTION_INTRASLAB

    # #: Required distance measure is hypocentral for in-slab events
    # REQUIRES_DISTANCES = {'rhypo'}

    # #: In-slab events require constraint of hypocentral depth and magnitude
    # REQUIRES_RUPTURE_PARAMETERS = {'mag', 'hypo_depth'}

    delta_c1 = -0.3
