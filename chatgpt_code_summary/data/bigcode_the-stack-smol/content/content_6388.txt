# coding: utf-8



def get_dict_output_dir_to_parameters_ini_dump_filename():
    import os
    dir_ = '.'
    output_dir_list = sorted([output_dir for output_dir in os.listdir(dir_) if output_dir.startswith('output')])
    ret = {}
    for output_dir in output_dir_list:
        with open(os.path.join(output_dir, 'parameters_ini_filename')) as f:
            parameters_ini_filename = list(f)[0].rstrip()
        ret[output_dir] = parameters_ini_filename + '.dump'
    return ret

        
dict_output_dir_to_parameters_ini_dump = get_dict_output_dir_to_parameters_ini_dump_filename()


import finess.util
import finess.params.util
import finess.dim2
import generate_iniparams


#     q(:, :, i - 1):
#     * i = 1: mass
#     * i = 2: momentum-1
#     * i = 3: momentum-2
#     * i = 4: momentum-3
#     * i = 5: energy
#     * i = 6: B1
#     * i = 7: B2
#     * i = 8: B3

import finess.viz.dim2





def L1_error_list(output_dir_list):
    
    global debug_B1_abs_error
    global debug_B2_abs_error
    global debug_B_perp, debug_B3, debug_u_perp, debug_u3
    global debug_B_perp_rel_error, debug_B_perp_abs_error, debug_B_perp_exact
    global debug_u_perp_rel_error, debug_u_perp_abs_error, debug_u_perp_exact
    global debug_B3_rel_error, debug_B3_abs_error, debug_B3_exact
    global debug_u3_rel_error, debug_u3_abs_error, debug_u3_exact
    global debug_B3_rel_error_100, debug_u3_rel_error_100
    global debug_tfinal
    global debug_B_plane_perp
    global debug_B_plane_perp_abs_error
    
    import finess.viz.dim2
    error_list = []
    for output_dir in output_dir_list:
        parameters_ini_dump_filename = dict_output_dir_to_parameters_ini_dump[output_dir]
        import os.path
        params = finess.params.util.read_params(os.path.join(output_dir, parameters_ini_dump_filename), generate_iniparams.parameter_list)
        xlow = params['grid', 'xlow']
        xhigh = params['grid', 'xhigh']
        ylow = params['grid', 'ylow']
        yhigh = params['grid', 'yhigh']
        mx = params['grid', 'mx']
        my = params['grid', 'my']
        dx = (xhigh - xlow) / float(mx)
        dy = (yhigh - ylow) / float(my)
        nout = params['finess', 'nout']

        tfinal, q, aux = finess.dim2.read_qa(params, nout)
        debug_tfinal = tfinal
        print "tfinal: ", tfinal
        from numpy import sin, cos, sum, abs, pi, max
        angle = params['initial', 'angle']
        X, Y = finess.viz.dim2.meshgrid(params)

        
        u3_exact = 0.1 * cos(2*pi * (X*cos(angle) + Y*sin(angle) + tfinal))
        
        B3_exact = u3_exact
        u_perp_exact = 0.1 * sin(2*pi * (X * cos(angle) + Y * sin(angle) + tfinal) )
        B_perp_exact = u_perp_exact
        
        rho_exact = 1.0
        u1_exact = -u_perp_exact * sin(angle)
        u2_exact = u_perp_exact * cos(angle)
        B1_exact = 1.0 * cos(angle) - B_perp_exact * sin(angle)
        B2_exact = 1.0 * sin(angle) + B_perp_exact * cos(angle)
        rho = q[:, :, 1 - 1]        
        u1 = q[:, :, 2 - 1] / q[:, :, 1 - 1]
        u2 = q[:, :, 3 - 1] / q[:, :, 1 - 1]
        u3 = q[:, :, 4 - 1] / q[:, :, 1 - 1]
        B1 = q[:, :, 6 - 1]
        B2 = q[:, :, 7 - 1]
        B3 = q[:, :, 8 - 1]
        u_perp = -u1 * sin(angle) + u2 * cos(angle)
        B_perp = -B1 * sin(angle) + B2 * cos(angle)
        
        
        L1_error_u_perp = sum(abs(u_perp - u_perp_exact))
        L1_u_perp_exact = sum(abs(u_perp_exact))
        
#        print "u_perp error: ", L1_error_u_perp / L1_u_perp_exact

        L1_error_u1 = sum(abs(u1 - u1_exact))
        L1_u1_exact = sum(abs(u1_exact))
        L1_error_u2 = sum(abs(u2 - u2_exact))
        L1_u2_exact = sum(abs(u2_exact))

        
        L1_error_u3 = sum(abs(u3 - u3_exact))
        L1_u3_exact = sum(abs(u3_exact))
#        print "u3 error: ", L1_error_u3 / L1_u3_exact

        
        L1_error_B_perp = sum(abs(B_perp - B_perp_exact))
        L1_B_perp_exact = sum(abs(B_perp_exact))
#        print "B_perp error: ", L1_error_B_perp / L1_B_perp_exact

        debug_B1_abs_error = abs(B1 - B1_exact)
        debug_B2_abs_error = abs(B2 - B2_exact)
        
        debug_B_perp_exact = B_perp_exact
        debug_B_perp_abs_error = abs(B_perp - B_perp_exact)
        debug_B_perp_rel_error = debug_B_perp_abs_error / abs(B_perp_exact)
        
        debug_u_perp_exact = u_perp_exact
        debug_u_perp_abs_error = abs(u_perp - u_perp_exact)
        debug_u_perp_rel_error = debug_u_perp_abs_error / abs(u_perp_exact)
        
        debug_B3_exact = B3_exact
        debug_B3_abs_error = abs(B3 - B3_exact)
        debug_B3_rel_error = debug_B3_abs_error / abs(B3_exact)
        debug_B3_rel_error_100 = debug_B3_rel_error * 100
        
        debug_u3_exact = u3_exact
        debug_u3_abs_error = abs(u3 - u3_exact)
        debug_u3_rel_error = debug_u3_abs_error / abs(u3_exact)
        debug_u3_rel_error_100 = 100 * debug_u3_rel_error
        
        debug_B3 = B3        
        debug_B_perp = B_perp
        debug_B_plane_perp = ((B3 / 0.1)**2 + (B_perp / 0.1)**2) * 0.1
        debug_B_plane_perp_abs_error = abs(debug_B_plane_perp - 0.1)
        

        L1_error_B3 = sum(abs(B3 - B3_exact))
        L1_B3_exact = sum(abs(B3_exact))
#        print "B3 error: ", L1_error_B3 / L1_B3_exact
        
#        delta = 0.25 * (L1_error_u_perp / L1_u_perp_exact + L1_error_u3 / L1_u3_exact + L1_error_B_perp / L1_B_perp_exact + L1_error_B3 / L1_B3_exact)
#        delta = 0.5 * (L1_error_B_perp / L1_B_perp_exact + L1_error_B3 / L1_B3_exact)
#        delta = 0.5 * (L1_error_u_perp / L1_u_perp_exact + L1_error_u3 / L1_u3_exact)
        #delta = max(abs(u3 - u3_exact))
        #delta = max(abs(u1 - u1_exact))
        #delta = max(abs(u2 - u2_exact))
        #delta = max(abs(u3 - u3_exact))
        #delta = max(abs(B1 - B1_exact))
        #delta = max(abs(B2 - B2_exact))
        #delta = max(abs(B3 - B3_exact))
        delta = max(abs(rho - rho_exact))
        #delta = L1_error_u1 / L1_u1_exact
        error_list.append(delta)
    return error_list


def log2_adjacent_ratio(error_list):
    order_list = []
    from numpy import log2
    for i in range(len(error_list) - 1):
        order_list.append(log2(error_list[i] / error_list[i+1]))
    return order_list

def L1_A_error_list(output_dir_list):
    from numpy import max
    global debug_A_abs_error    
    
    import finess.viz.dim2
    error_list = []
    for output_dir in output_dir_list:
        parameters_ini_dump_filename = dict_output_dir_to_parameters_ini_dump[output_dir]
        import os.path
        params = finess.params.util.read_params(os.path.join(output_dir, parameters_ini_dump_filename), generate_iniparams.parameter_list)
        xlow = params['grid', 'xlow']
        xhigh = params['grid', 'xhigh']
        ylow = params['grid', 'ylow']
        yhigh = params['grid', 'yhigh']
        mx = params['grid', 'mx']
        my = params['grid', 'my']
        dx = (xhigh - xlow) / float(mx)
        dy = (yhigh - ylow) / float(my)
        nout = params['finess', 'nout']

        tfinal, q, aux = finess.dim2.read_qa(params, nout)
        A = aux[:, :, 1 - 1]        
        
        from numpy import allclose, sin, cos, sum, abs, pi
        angle = params['initial', 'angle']

        
        X, Y = finess.viz.dim2.meshgrid(params)
        
        A_exact = -X * sin(angle) + Y * cos(angle) + 0.1 / (2 * pi) * cos(2*pi * (X*cos(angle) + Y*sin(angle) + tfinal))
        
        debug_A_abs_error = abs(A - A_exact)
        L1_A_exact = sum(abs(A_exact))
        L1_A_error = sum(abs(A - A_exact))
        #delta = L1_A_error / L1_A_exact
        delta = max(abs(A - A_exact))
        error_list.append(delta)
    return error_list




#output_dir_list = ['output_1deg_%(i)02d' % {'i': i} for i in range(6)]
#error_list = L1_error_list(output_dir_list)
#order_list = log2_adjacent_ratio(error_list)
#print order_list
#
#
#output_dir_list = ['output_30deg_%(i)02d' % {'i': i} for i in range(6)]
#error_list = L1_error_list(output_dir_list)
#order_list = log2_adjacent_ratio(error_list)
#print order_list
#
#
## In[140]:

output_dir_list = ['output_30deg_%(i)02d' % {'i': i} for i in [0, 1, 2, 3, 4]]
error_list = L1_error_list(output_dir_list)
order_list = log2_adjacent_ratio(error_list)
print 'rho'
print order_list
print error_list



A_error_list = L1_A_error_list(output_dir_list)
A_order_list = log2_adjacent_ratio(A_error_list)
print 'A:'
print A_order_list
print A_error_list

