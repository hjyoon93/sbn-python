#the problem is conditional probability table is not updated with beta distributions. try to use pypgm instead of bnlearn
from pgmpy.models import BayesianNetwork

import numpy as np
import pandas as pd
import math
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ApproxInference
import statistics
import ast

from pgmpy.inference import VariableElimination
import bnlearn as bn


#translate absolute counts to opinion's belief
def translateB(r,s):
    return r/(2+r+s)

#translate absolute counts to opinion's disbelief
def translateD(r,s):
    return s/(2+r+s)

#translate absolute counts to opinion's uncertainty
def translateU(r,s):
    return 2/(2+r+s)


#translate opinion to absolute counts
def translateR(b,u):
    if u != 0:
        return (2*b)/u
    else:
        positiveInf = math.inf
        return b * positiveInf

with open("variables.txt", "r+") as file1:
    # Reading from a file
    variables_0 = file1.readlines()
    print(variables_0)

variables = []
for sub in variables_0:
    variables.append(sub.replace("\n", ""))
print(variables)
print(len(variables))


with open("evidence.txt", "r+") as file1:
    # Reading from a file
    evidence_0 = file1.readlines()
    print(evidence_0)

evidences_1 = []
for sub in evidence_0:
    evidences_1.append(sub.replace("\n", ""))
print(evidences_1)
print(len(evidences_1))

evidences = []
for i in range(len(evidences_1)):
    evidences.append(ast.literal_eval(evidences_1[i]))
print(evidences)
print(len(evidences))

count1 = 0
count_x = 0
for times in range(100):

    iter_converge = 0
    count = 0
    b = []
    d = []
    u = []
    count_inner = 0
    count_inner1 = 0
    b_exact = []
    d_exact = []
    u_exact = []
    diff_list = []
    model = bn.import_DAG('sprinkler')

    model1 = BayesianNetwork([('Rain', 'Wet_Grass'), ('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler'), ('Sprinkler', 'Wet_Grass')])

    #####
    model4 = BayesianNetwork([('Rain', 'Wet_Grass'), ('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler'), ('Sprinkler', 'Wet_Grass')])
    ####

    #print(model1)
    #exit()
    #sampling 1000000 rows
    #df = bn.sampling(model, n=60)
    df = pd.read_csv('test100.csv')
    #print(df)

    #counts for cloudy
    cloudy = []
    #cloudy
    a = df.loc[:,"Cloudy"]
    for i in a:
        #print(i)
        cloudy.append(i)
    count_cloud = {'f': '', 't': ''}
    count_cloud['f'] = cloudy.count(0)
    count_cloud['t'] = cloudy.count(1)
    print('counts for cloudy:', count_cloud)
    #counts for cloudy


    #counts for sprinkler
    sprinkler = []

    a = df.loc[:,"Sprinkler"]
    for i in a:

        sprinkler.append(i)
    count_sprinkler = {'f': '', 't': ''}
    count_sprinkler['f'] = sprinkler.count(0)
    count_sprinkler['t'] = sprinkler.count(1)
    print('counts for sprinkler:', count_sprinkler)
    #counts for sprinkler


    ##############counts for rain
    rain = []

    a = df.loc[:, "Rain"]
    for i in a:
        # print(i)
        rain.append(i)
    count_rain = {'f': '', 't': ''}
    count_rain['f'] = rain.count(0)
    count_rain['t'] = rain.count(1)
    print('counts for rain:', count_rain)
    ##############counts for rain



    #counts for wet_grass
    wet_grass = []

    a = df.loc[:,"Wet_Grass"]
    for i in a:

        wet_grass.append(i)
    count_wet_grass = {'f': '', 't': ''}
    count_wet_grass['f'] = wet_grass.count(0)
    count_wet_grass['t'] = wet_grass.count(1)
    print('counts for wet_grass:', count_wet_grass)
    #counts for wet_grass



    #counts for conditionals
    sp_cl_tf = 0
    sp_cl_tt = 0
    sp_cl_ft = 0
    sp_cl_ff = 0

    ra_cl_tf = 0
    ra_cl_tt = 0
    ra_cl_ft = 0
    ra_cl_ff = 0

    we_sp_ra_f_ff = 0
    we_sp_ra_t_ff = 0
    we_sp_ra_f_ft = 0
    we_sp_ra_t_ft = 0

    we_sp_ra_f_tf = 0
    we_sp_ra_t_tf = 0
    we_sp_ra_f_tt = 0
    we_sp_ra_t_tt = 0


    for i in range(100):
        first = df.loc[i]
        print(first)

        cl = first['Cloudy']
        sp = first['Sprinkler']
        ra = first['Rain']
        we = first['Wet_Grass']

        if sp == 1 and cl == 1:
            sp_cl_tt += 1

        if sp == 0 and cl == 1:
            sp_cl_ft += 1

        if sp == 1 and cl == 0:
            sp_cl_tf += 1

        if sp == 0 and cl == 0:
            sp_cl_ff += 1

        if we == 0 and sp == 0 and ra == 0:
            we_sp_ra_f_ff += 1

        if we == 1 and sp == 0 and ra == 0:
            we_sp_ra_t_ff += 1

        if we == 0 and sp == 0 and ra == 1:
            we_sp_ra_f_ft += 1

        if we == 1 and sp == 0 and ra == 1:
            we_sp_ra_t_ft += 1

        if we == 0 and sp == 1 and ra == 0:
            we_sp_ra_f_tf += 1

        if we == 1 and sp == 1 and ra == 0:
            we_sp_ra_t_tf += 1

        if we == 0 and sp == 1 and ra == 1:
            we_sp_ra_f_tt += 1

        if we == 1 and sp == 1 and ra == 1:
            we_sp_ra_t_tt += 1

        if ra == 1 and cl == 1:
            ra_cl_tt += 1

        if ra == 1 and cl == 0:
            ra_cl_tf += 1

        if ra == 0 and cl == 1:
            ra_cl_ft += 1

        if ra == 0 and cl == 0:
            ra_cl_ff += 1

    # count for sprinkler given cloudy
    count_sp_cl = {'f|f': '', 't|f': '', 'f|t': '', 't|t': ''}
    count_sp_cl['f|f'] = sp_cl_ff
    count_sp_cl['t|f'] = sp_cl_tf
    count_sp_cl['f|t'] = sp_cl_ft
    count_sp_cl['t|t'] = sp_cl_tt
    # count for sprinkler given cloudy


    # count for rain given cloudy
    count_ra_cl = {'f|f': '', 't|f': '', 'f|t': '', 't|t': ''}
    count_ra_cl['f|f'] = ra_cl_ff
    count_ra_cl['t|f'] = ra_cl_tf
    count_ra_cl['f|t'] = ra_cl_ft
    count_ra_cl['t|t'] = ra_cl_tt
    # count for rain given cloudy


    # count for wet_grass given sprinkler and rain
    count_we_sp_ra = {'f|ff': '', 't|ff': '', 'f|ft': '', 't|ft': '', 'f|tf': '', 't|tf': '', 'f|tt': '', 't|tt': ''}
    count_we_sp_ra['f|ff'] = we_sp_ra_f_ff
    count_we_sp_ra['t|ff'] = we_sp_ra_t_ff
    count_we_sp_ra['f|ft'] = we_sp_ra_f_ft
    count_we_sp_ra['t|ft'] = we_sp_ra_t_ft
    count_we_sp_ra['f|tf'] = we_sp_ra_f_tf
    count_we_sp_ra['t|tf'] = we_sp_ra_t_tf
    count_we_sp_ra['f|tt'] = we_sp_ra_f_tt
    count_we_sp_ra['t|tt'] = we_sp_ra_t_tt
    # count for wet_grass given sprinkler and rain


    #initializing mean and variance
    sum_0 = []
    sum_1 = []
    var = []

    sum_0_exact = []
    sum_1_exact = []
    var_exact = []

    #initializing bn inference



    #inp = int(input("Enter user inputs for number of iterations : ") or "1000")

    #stochastic process 1000 times
    for i in range(20):
        #print("input:", inp)
        print("number of current iteration:", i)
        count += 1
        #sampling beta distribution for cloud
        cloudy_beta_f = np.random.beta(count_cloud['f'] + 1, count_cloud['t'] + 1)
        cloudy_beta_t = 1 - cloudy_beta_f

        #printing beta distribution for cloud
        print("beta f for cloudy:",cloudy_beta_f)
        print("beta t for cloudy:",cloudy_beta_t)

        # Defining individual CPDs for cloud.
        cpd_cloudy = TabularCPD(variable='Cloudy', variable_card=2, values=[[cloudy_beta_f], [cloudy_beta_t]])
        print(cpd_cloudy)



        #sampling beta distribution for sprinkler
        sprinkler_beta_f = np.random.beta(count_sprinkler['f'] + 1, count_sprinkler['t'] + 1)
        sprinkler_beta_t = 1 - sprinkler_beta_f

        #printing beta distribution for sprinkler
        print("beta f for sprinkler:",sprinkler_beta_f)
        print("beta t for sprinkler:",sprinkler_beta_t)



        #sampling beta distribution for rain
        rain_beta_f = np.random.beta(count_rain['f'] + 1, count_rain['t'] + 1)
        rain_beta_t = 1 - rain_beta_f

        #printing beta distribution for rain
        print("beta f for rain:",rain_beta_f)
        print("beta t for rain:",rain_beta_t)



        #sampling beta distribution for wet_grass
        wet_grass_beta_f = np.random.beta(count_wet_grass['f'] + 1, count_wet_grass['t'] + 1)
        wet_grass_beta_t = 1 - wet_grass_beta_f

        #printing beta distribution for wet_grass
        print("beta f for wet_grass:",wet_grass_beta_f)
        print("beta t for wet_grass:",wet_grass_beta_t)




        sp_cl_ff_beta = np.random.beta(sp_cl_ff+1, sp_cl_tf+1)
        sp_cl_tf_beta = 1-sp_cl_ff_beta

        print("sp_cl_ff_beta:", sp_cl_ff_beta)
        print("sp_cl_tf_beta:",sp_cl_tf_beta)
        print(sp_cl_ff_beta+sp_cl_tf_beta)

        sp_cl_ft_beta = np.random.beta(sp_cl_ft+1, sp_cl_tt+1)
        sp_cl_tt_beta = 1 - sp_cl_ft_beta


        cpd_sprinkler = TabularCPD(variable='Sprinkler', variable_card=2,
                                   values=[[sp_cl_ff_beta, sp_cl_ft_beta],
                                           [sp_cl_tf_beta, sp_cl_tt_beta]],
                                   evidence=['Cloudy'],
                                   evidence_card=[2])


        ####################################
        ra_cl_ff_beta = np.random.beta(ra_cl_ff + 1, ra_cl_tf + 1)
        ra_cl_tf_beta = 1 - ra_cl_ff_beta

        print("ra_cl_ff_beta:",ra_cl_ff_beta)
        print("ra_cl_tf_beta:",ra_cl_tf_beta)
        print(ra_cl_ff_beta + ra_cl_tf_beta)

        ra_cl_ft_beta = np.random.beta(ra_cl_ft + 1, ra_cl_tt + 1)
        ra_cl_tt_beta = 1 - ra_cl_ft_beta

        print("ra_cl_ft_beta:",ra_cl_ft_beta)
        print("ra_cl_tt_beta:",ra_cl_tt_beta)
        print(ra_cl_ft_beta + ra_cl_tt_beta)

        cpd_rain = TabularCPD(variable='Rain', variable_card=2,
                              values=[[ra_cl_ff_beta, ra_cl_ft_beta],
                                      [ra_cl_tf_beta, ra_cl_tt_beta]],
                              evidence=['Cloudy'],
                              evidence_card=[2])



        we_sp_ra_f_ff_beta = np.random.beta(we_sp_ra_f_ff + 1, we_sp_ra_t_ff + 1)
        we_sp_ra_t_ff_beta = 1 - we_sp_ra_f_ff_beta

        print("we_sp_ra_f_ff_beta:",we_sp_ra_f_ff_beta)
        print("we_sp_ra_t_ff_beta:",we_sp_ra_t_ff_beta)
        print(we_sp_ra_f_ff_beta + we_sp_ra_t_ff_beta)

        we_sp_ra_f_ft_beta = np.random.beta(we_sp_ra_f_ft + 1, we_sp_ra_t_ft + 1)
        we_sp_ra_t_ft_beta = 1 - we_sp_ra_f_ft_beta

        print("we_sp_ra_f_ft_beta:",we_sp_ra_f_ft_beta)
        print("we_sp_ra_t_ft_beta:",we_sp_ra_t_ft_beta)
        print(we_sp_ra_f_ft_beta + we_sp_ra_t_ft_beta)

        we_sp_ra_f_tf_beta = np.random.beta(we_sp_ra_f_tf + 1, we_sp_ra_t_tf + 1)
        we_sp_ra_t_tf_beta = 1 - we_sp_ra_f_tf_beta

        print("we_sp_ra_f_tf_beta:",we_sp_ra_f_tf_beta)
        print("we_sp_ra_t_tf_beta:",we_sp_ra_t_tf_beta)
        print(we_sp_ra_f_tf_beta + we_sp_ra_t_tf_beta)

        we_sp_ra_f_tt_beta = np.random.beta(we_sp_ra_f_tt + 1, we_sp_ra_t_tt + 1)
        we_sp_ra_t_tt_beta = 1 - we_sp_ra_f_tt_beta

        print("we_sp_ra_f_tt_beta:",we_sp_ra_f_tt_beta)
        print("we_sp_ra_t_tt_beta:",we_sp_ra_t_tt_beta)
        print(we_sp_ra_f_tt_beta + we_sp_ra_t_tt_beta)

        cpd_wet_grass = TabularCPD(variable='Wet_Grass', variable_card=2,
                           values=[[we_sp_ra_f_ff_beta,we_sp_ra_f_tf_beta ,we_sp_ra_f_ft_beta ,we_sp_ra_f_tt_beta],
                                   [we_sp_ra_t_ff_beta,we_sp_ra_t_tf_beta ,we_sp_ra_t_ft_beta ,we_sp_ra_t_tt_beta]],
                           evidence=['Sprinkler', 'Rain'],
                           evidence_card=[2, 2])

        #print(cpd_wet_grass.values)
        #quit()
        # Associating the CPDs with the network
        model1.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler,  cpd_wet_grass)
    ######
    #######
        # check_model checks for the network structure and CPDs and verifies that the CPDs are correctly
        # defined and sum to 1.
        #print(model1.check_model())
        print([count_cloud['f'], count_cloud['t']])
        print([[sp_cl_ff, sp_cl_ft],
                                           [sp_cl_tf, sp_cl_tt]])
        print([[ra_cl_ff, ra_cl_ft],
                                      [ra_cl_tf, ra_cl_tt]])
        print([[we_sp_ra_f_ff,we_sp_ra_f_tf ,we_sp_ra_f_ft ,we_sp_ra_f_tt],
                                   [we_sp_ra_t_ff,we_sp_ra_t_tf ,we_sp_ra_t_ft ,we_sp_ra_t_tt]])
        #quit()


        #infer = VariableElimination(model1)

        #q_1 = infer.query(variables=['Rain'], evidence={'Wet_Grass': 1, "Cloudy": 0, "Sprinkler":1})
        from pgmpy.sampling import BayesianModelSampling
        ###### bayesianmodel sampling
        """
        inference1 = BayesianModelSampling(model1)
        aa = inference1.forward_sample(size=25)

        print(aa)
        print(aa['Rain'])
        print(list(aa['Rain']))
        #print(len(list[aa['Rain']]))
        cloudy11 = list(aa['Cloudy'])
        rain11 = list(aa['Rain'])
        sprinkler11 = list(aa['Sprinkler'])
        wetgrass11 = list(aa['Wet_Grass'])

        dataset1 = []
        for i in range(25):
            dataset1.append({'Rain': rain11[i], 'Wet_Grass': wetgrass11[i], 'Cloudy': cloudy11[i]})
        print("dataset1:",dataset1)

        f = open("evidence.txt", "a")
        for i in range(25):
            f.write(str(dataset1[i]) + '\n')
        f.close()
        quit()
        """

        infer = ApproxInference(model1)
        inference = BayesianModelSampling(model1)

        from pgmpy.factors.discrete import State
        keys =[]
        items = []
        print(evidences[times])
        for key in evidences[times]:
            print(key)
            keys.append(key)
            print(evidences[times][key])
            items.append(evidences[times][key])

        evidence_list = []

        for i in range(len(keys)):
            evidence_list.append(State(var=keys[i], state=items[i]))

        print(evidence_list)
        #quit()


        #evidence_list = [State(var='Cloudy', state=0), State(var='Wet_Grass', state=1)]
        sampledata = inference.rejection_sample(evidence=evidence_list, size=150)
        #sampledata = inference.forward_sample(size=10000)
        #sampledata = inference.likelihood_weighted_sample(evidence=evidence_list,size=100)
        #print(sampledata)
        #quit()


        #print(sampledata)
        #df11 = bn.sampling(model, n=1000, methodtype='bayes')
        #print(df11)
        #quit()



        #print(approx_query2)
        #quit()


        ######
        model4.add_cpds(cpd_cloudy, cpd_rain, cpd_sprinkler,  cpd_wet_grass)
        #print(model4.get_cpds())
        #quit()
        infer1 = VariableElimination(model4)

        #quit()

        approx_query2 = infer.query(variables=[variables[times]], evidence=evidences[times], samples=sampledata)
        q_1 = infer1.query(variables=[variables[times]], evidence=evidences[times])
        #######
        print(approx_query2)
        #print(approx_query2.get_value(Wet_Grass=1))
        print(q_1)
        #quit()
        #quit()
        #quit()
        #print(q_1.get_value(Wet_Grass=1))
        #quit()
        #print(q1)
        #print(q1.get_value(Wet_Grass=1))
        #print(q_1.get_value(Wet_Grass=1))

        #print("rain_false:", q1.values[0])
        #print(1-q1.values[0])
        if variables[times] == "Rain":

            if len(approx_query2.state_names) == 1 and approx_query2.state_names['Rain'][0] == 0:
                rain_false = approx_query2.get_value(Rain=0)
                rain_true = 1.0 - rain_false

            elif len(approx_query2.state_names) == 1 and approx_query2.state_names['Rain'][0] == 1:
                rain_true = approx_query2.get_value(Rain=1)
                rain_false = 1.0 - rain_true

            else:
                rain_false = approx_query2.get_value(Rain=0)
                rain_true = approx_query2.get_value(Rain=1)

            print("rain_false:", rain_false)
            print("rain_true:", rain_true)

            print(q_1.get_value(Rain=0))
            print(q_1.get_value(Rain=1))

            diff_list.append(abs(rain_true - q_1.get_value(Rain=1)))
            diff1 = abs(rain_true - q_1.get_value(Rain=1))
            x = statistics.mean(diff_list)
            print("diff:", x)
            sum_0.append(rain_false)
            sum_1.append(rain_true)

            sum_0_exact.append(q_1.get_value(Rain=0))
            sum_1_exact.append(q_1.get_value(Rain=1))

        elif variables[times] == "Sprinkler":
            if len(approx_query2.state_names) == 1 and approx_query2.state_names['Sprinkler'][0] == 0:
                spr_false = approx_query2.get_value(Sprinkler=0)
                spr_true = 1.0 - spr_false

            elif len(approx_query2.state_names) == 1 and approx_query2.state_names['Sprinkler'][0] == 1:
                spr_true = approx_query2.get_value(Sprinkler=1)
                spr_false = 1.0 - spr_true
            else:
                spr_false = approx_query2.get_value(Sprinkler=0)
                spr_true = approx_query2.get_value(Sprinkler=1)

            print("rain_false:", spr_false)
            print("rain_true:", spr_true)

            print(q_1.get_value(Sprinkler=0))
            print(q_1.get_value(Sprinkler=1))

            diff_list.append(abs(spr_true - q_1.get_value(Sprinkler=1)))
            diff1 = abs(spr_true - q_1.get_value(Sprinkler=1))
            x = statistics.mean(diff_list)
            print("diff:", x)
            sum_0.append(spr_false)
            sum_1.append(spr_true)

            sum_0_exact.append(q_1.get_value(Sprinkler=0))
            sum_1_exact.append(q_1.get_value(Sprinkler=1))

        elif variables[times] == "Cloudy":
            if len(approx_query2.state_names) == 1 and approx_query2.state_names['Cloudy'][0] == 0:
                cl_false = approx_query2.get_value(Cloudy=0)
                cl_true = 1.0 - cl_false

            elif len(approx_query2.state_names) == 1 and approx_query2.state_names['Cloudy'][0] == 1:
                cl_true = approx_query2.get_value(Cloudy=1)
                cl_false = 1.0 - cl_true


            else:
                cl_false = approx_query2.get_value(Cloudy=0)
                cl_true = approx_query2.get_value(Cloudy=1)

            print("rain_false:", cl_false)
            print("rain_true:", cl_true)

            print(q_1.get_value(Cloudy=0))
            print(q_1.get_value(Cloudy=1))

            diff_list.append(abs(cl_true - q_1.get_value(Cloudy=1)))
            diff1 = abs(cl_true - q_1.get_value(Cloudy=1))
            x = statistics.mean(diff_list)
            print("diff:", x)
            sum_0.append(cl_false)
            sum_1.append(cl_true)

            sum_0_exact.append(q_1.get_value(Cloudy=0))
            sum_1_exact.append(q_1.get_value(Cloudy=1))

        elif variables[times] == "Wet_Grass":
            print(approx_query2.state_names)
            print(type(approx_query2.state_names))

            print(approx_query2.state_names['Wet_Grass'][0])

            if len(approx_query2.state_names) == 1 and approx_query2.state_names['Wet_Grass'][0] == 0:
                wg_false = approx_query2.get_value(Wet_Grass=0)
                wg_true = 1.0 - wg_false

            elif len(approx_query2.state_names) == 1 and approx_query2.state_names['Wet_Grass'][0] == 1:
                wg_true = approx_query2.get_value(Wet_Grass=1)
                wg_false = 1.0 - wg_true

            else:
                wg_false = approx_query2.get_value(Wet_Grass=0)
                wg_true = approx_query2.get_value(Wet_Grass=1)

            print("rain_false:", wg_false)
            print("rain_true:", wg_true)

            print(q_1.get_value(Wet_Grass=0))
            print(q_1.get_value(Wet_Grass=1))

            diff_list.append(abs(wg_true - q_1.get_value(Wet_Grass=1)))
            diff1 = abs(wg_true - q_1.get_value(Wet_Grass=1))
            x = statistics.mean(diff_list)
            print("diff:", x)
            sum_0.append(wg_false)
            sum_1.append(wg_true)

            sum_0_exact.append(q_1.get_value(Wet_Grass=0))
            sum_1_exact.append(q_1.get_value(Wet_Grass=1))


            #############################
        print("length of len sum_0:", len(sum_0))
        print("length of len sum_1:", len(sum_1))
        mean_f = sum(sum_0) / len(sum_0)
        mean_t = sum(sum_1) / len(sum_1)
        var_0 = 0
        var_1 = 0
        for j in range(len(sum_0)):
            var_0 += (sum_0[j] - mean_f) ** 2

        for a in range(len(sum_1)):
            var_1 += (sum_1[a] - mean_t) ** 2

        if count == 1:
            var_f = 0
            var_t = 0
        else:
            var_f = var_0 / (len(sum_0) - 1)  # (len(sum_0)-1)

            var_t = var_1 / (len(sum_1) - 1)  # (len(sum_0)-1)

        print("mean_f:", mean_f)
        print("mean_t:", mean_t)

        print("var_f:", var_f)
        print("var_t:", var_t)


        def posterior(mu, var):
            if var == 0:
                posterior_beta = 0
            else:
                posterior_beta = (((mu * (1 - mu)) / var) - 1) * mu
            return posterior_beta


        # if count > 1:

        alpha = posterior(mean_t, var_t)
        beta = posterior(mean_f, var_f)

        print("alpha:", alpha)
        print("beta:", beta)

        r_output = alpha - 1
        s_output = beta - 1

        print("r:", r_output)
        print("s:", s_output)

        adding = 0
        adding1 = 0
        adding2 = 0

        if r_output < 0:
            adding1 = abs(r_output)

        if s_output < 0:
            adding2 = abs(s_output)

        if adding1 > adding2:
            adding = adding1
        else:
            adding = adding2

        if r_output < 0 and s_output < 0:
            if r_output > s_output:
                adding = abs(s_output)
            elif r_output < s_output:
                adding = abs(r_output)

        r_output = r_output + adding
        s_output = s_output + adding

        print("r:", r_output)
        print("s:", s_output)

        print("b:", translateB(r_output, s_output))
        print("d:", translateD(r_output, s_output))
        print("u:", translateU(r_output, s_output))

        b.append(translateB(r_output, s_output))
        d.append(translateD(r_output, s_output))
        u.append(translateU(r_output, s_output))


        #####
        #############################
        print("length of len sum_0:", len(sum_0_exact))
        print("length of len sum_1:", len(sum_1_exact))
        mean_f_exact = sum(sum_0_exact) / len(sum_0_exact)
        mean_t_exact = sum(sum_1_exact) / len(sum_1_exact)
        var_0_exact = 0
        var_1_exact = 0
        for j in range(len(sum_0_exact)):
            var_0_exact += (sum_0_exact[j] - mean_f_exact) ** 2

        for a in range(len(sum_1_exact)):
            var_1_exact += (sum_1_exact[a] - mean_t_exact) ** 2

        if count == 1:
            var_f_exact = 0
            var_t_exact = 0
        else:
            var_f_exact = var_0_exact / (len(sum_0_exact) - 1)  # (len(sum_0)-1)

            var_t_exact = var_1_exact / (len(sum_1_exact) - 1)  # (len(sum_0)-1)

        print("mean_f:", mean_f_exact)
        print("mean_t:", mean_t_exact)

        print("var_f:", var_f_exact)
        print("var_t:", var_t_exact)




        # if count > 1:

        alpha_exact = posterior(mean_t_exact, var_t_exact)
        beta_exact = posterior(mean_f_exact, var_f_exact)

        print("alpha:", alpha_exact)
        print("beta:", beta_exact)

        r_output_exact = alpha_exact - 1
        s_output_exact = beta_exact - 1



        adding_exact = 0
        adding1_exact = 0
        adding2_exact = 0

        if r_output_exact < 0:
            adding1_exact = abs(r_output_exact)

        if s_output_exact < 0:
            adding2_exact = abs(s_output_exact)

        if adding1_exact > adding2_exact:
            adding_exact = adding1_exact
        else:
            adding_exact = adding2_exact

        if r_output_exact < 0 and s_output_exact < 0:
            if r_output_exact > s_output_exact:
                adding_exact = abs(s_output_exact)
            elif r_output_exact < s_output_exact:
                adding_exact = abs(r_output_exact)

        r_output_exact = r_output_exact + adding_exact
        s_output_exact = s_output_exact + adding_exact



        print("b:", translateB(r_output_exact, s_output_exact))
        print("d:", translateD(r_output_exact, s_output_exact))
        print("u:", translateU(r_output_exact, s_output_exact))

        b_exact.append(translateB(r_output_exact, s_output_exact))
        d_exact.append(translateD(r_output_exact, s_output_exact))
        u_exact.append(translateU(r_output_exact, s_output_exact))
        ####

        #if count <= 45:
        last_bms_b = b[-1]
        last_bms_d = d[-1]
        last_bms_u = u[-1]

        last_exact_b = b_exact[-1]
        last_exact_d = d_exact[-1]
        last_exact_u = u_exact[-1]



        last_b = abs(last_exact_b - last_bms_b)
        last_d = abs(last_exact_d - last_bms_d)
        last_u = abs(last_exact_u - last_bms_u)

        """
        if len(b) % 5 == 0:
            if abs(b[-1]-b[-2]) < 0.005 and abs(d[-1]-d[-2]) < 0.005 and abs(u[-1]-u[-2] < 0.005):
                iter_converge += 1
                print(translateB(r_output, s_output) + translateD(r_output, s_output) + translateU(r_output, s_output))
                print("sbn_expert:", translateD(r_output, s_output) + translateU(r_output, s_output))
                print("sbn_no_knowledge:", translateD(r_output, s_output) + (translateU(r_output, s_output) / 2))
                print("number of iterations finished:", count)
                print(b[-1])
                print(b[-2])
                print("the different:",abs(b[-1]-b[-2]))
                if iter_converge == 5:
                #    print("iter_converge:", iter_converge)
                    break
            else:
                print("sbn_expert:", translateD(r_output, s_output) + translateU(r_output, s_output))
                print("sbn_no_knowledge:", translateD(r_output, s_output) + (translateU(r_output, s_output) / 2))
                iter_converge = 0
        """

        """
        if count >= 45:
            if len(b_exact) % 5 == 0:
                if abs(b_exact[-1] - b_exact[-2]) < 0.005 and abs(d_exact[-1] - d_exact[-2]) < 0.005 and abs(u_exact[-1] - u_exact[-2] < 0.005):
                    iter_converge += 1
                    #print(translateB(r_output, s_output) + translateD(r_output, s_output) + translateU(r_output, s_output))
                    #print("sbn_expert:", translateD(r_output, s_output) + translateU(r_output, s_output))
                    #print("sbn_no_knowledge:", translateD(r_output, s_output) + (translateU(r_output, s_output) / 2))
                    print("number of iterations finished:", count)
                    #print(b[-1])
                    #print(b[-2])
                    #print("the different:", abs(b[-1] - b[-2]))
                    if iter_converge == 5:
                        #    print("iter_converge:", iter_converge)
                        break
                else:
                    #print("sbn_expert:", translateD(r_output, s_output) + translateU(r_output, s_output))
                    #print("sbn_no_knowledge:", translateD(r_output, s_output) + (translateU(r_output, s_output) / 2))
                    iter_converge = 0

        """



        """
            
            if abs(b_exact[-1]-b_exact[-2]) < 0.001 and abs(d_exact[-1]-d_exact[-2]) < 0.001 and abs(u_exact[-1]-u_exact[-2] < 0.001):
                count_inner1 += 1
                print("counter_inner1:", count_inner1)
                if count_inner1 == 1:
                    print(translateB(r_output_exact, s_output_exact) + translateD(r_output_exact, s_output_exact) + translateU(r_output_exact, s_output_exact))
                    print("sbn_expert_exact:", translateD(r_output_exact, s_output_exact) + translateU(r_output_exact, s_output_exact))
                    print("sbn_no_knowledge_exact:", translateD(r_output_exact, s_output_exact) + (translateU(r_output_exact, s_output_exact) / 2))
                    print("number of iterations finished:", count)
                    print("exact:",b_exact[-1])
                    print("approx:",b[-1])
                    print("b:", b)
                    print("b_exact", b_exact)
                    #print(b_exact[-2])
                    print("the different:",abs(b_exact[-1]-b_exact[-2]))
                    last_exact = b_exact[-1]
                    print("last_exact:", last_exact)
                    if count_inner1 == 1 and count_inner >= 1:
                        print("countinner:", count_inner)
                        print("countinner1:", count_inner1)
                        break


        """


            #else:
            #    print("sbn_expert:", translateD(r_output, s_output) + translateU(r_output, s_output))
            #    print("sbn_no_knowledge:", translateD(r_output, s_output) + (translateU(r_output, s_output) / 2))






    #if x <= 0.1:
    f = open("diff_sbn_bms_b.txt", "a")
    f.write(str(last_b) + '\n')
    f.close()

    #f = open("diff_bms.txt", "a")
    #f.write(str(x) + '\n')
    #f.close()

    #f = open("iteration_bms.txt", "a")
    #f.write(str(count) + '\n')
    #f.close()






"""
    if last_b < 0.1 and count != 1000:
        count1 += 1

        f = open("diff_sbn_bms_b.txt", "a")
        f.write(str(last_b)+'\n')
        f.close()

        #last1 = abs(last_exact_d - last_bms_d)
        #f = open("diff_sbn_bms_d.txt", "a")
        #f.write(str(last1) + '\n')
        #f.close()

        #last2 = abs(last_exact_u - last_bms_u)
        #f = open("diff_sbn_bms_u.txt", "a")
        #f.write(str(last2) + '\n')
        #f.close()

        f = open("diff_bms.txt", "a")
        f.write(str(x)+'\n')
        f.close()

        f = open("iteration_bms.txt", "a")
        f.write(str(count) + '\n')
        f.close()

        if count1 == 20:

            with open("diff_bms.txt", "r+") as file1:
                # Reading from a file
                content = file1.readlines()
                #print(content)

            for i in range(len(content)):
                content[i] = float(content[i])

            #content = content[-50:]

            import statistics

            #print(content)

            avg_bms = statistics.mean(content)

            f = open("bms_diff_avg.txt", "a")
            f.write(str(avg_bms) + '\n')
            f.close()

            with open("diff_sbn_bms_b.txt", "r+") as file1:
                # Reading from a file
                content = file1.readlines()
                #print(content)

            for i in range(len(content)):
                content[i] = float(content[i])

            #content = content[-50:]

            import statistics

            # print(content)

            avg_bms_sbn = statistics.mean(content)

            f = open("bms_sbn_avg.txt", "a")
            f.write(str(avg_bms_sbn) + '\n')
            f.close()

            exit()
        #file_to_delete = open("diff_bms.txt", 'w')
        #file_to_delete.close()

        #file_to_delete = open("diff_sbn_bms_b.txt", 'w')
        #file_to_delete.close()


            #f = open('diff_bms.txt', 'r+')
            #f.truncate(0)

            #f = open('diff_sbn_bms_b.txt', 'r+')
            #f.truncate(0)
"""
#############################
""""
print("length of len sum_0:",len(sum_0))
print("length of len sum_1:",len(sum_1))
mean_f = sum(sum_0)/len(sum_0)
mean_t = sum(sum_1)/len(sum_1)
var_0 = 0
var_1 = 0
for j in range(len(sum_0)):
    var_0 += (sum_0[j]-mean_f)**2

for a in range(len(sum_1)):
    var_1 += (sum_1[a]-mean_t)**2

var_f = var_0/(len(sum_0)-1)

var_t = var_1/(len(sum_1)-1)


print(mean_f)
print(mean_t)

print(var_f)
print(var_t)


def posterior(mu, var):
    posterior_beta = (((mu*(1-mu))/var)-1)*mu
    return posterior_beta

alpha = posterior(mean_t, var_t)
beta = posterior(mean_f, var_f)

print("alpha:",alpha)
print("beta:",beta)

r_output = alpha-1
s_output = beta-1

print("r:", r_output)
print("s:", s_output)

adding = 0
adding1 = 0
adding2 = 0

if r_output < 0:
    adding1 = abs(r_output)

if s_output < 0:
    adding2 = abs(s_output)

if adding1 > adding2:
    adding = adding1
else:
    adding = adding2


if r_output < 0 and s_output < 0:
    if r_output > s_output:
        adding = abs(s_output)
    elif r_output < s_output:
        adding = abs(r_output)

r_output = r_output + adding
s_output = s_output + adding

print("r:", r_output)
print("s:", s_output)

print("b:",translateB(r_output,s_output))
print("d:",translateD(r_output,s_output))
print("u:",translateU(r_output,s_output))

print(translateB(r_output,s_output)+translateD(r_output,s_output)+translateU(r_output,s_output))
print("sbn_expert:",translateD(r_output,s_output)+translateU(r_output,s_output))
print("sbn_no_knowledge:",translateD(r_output,s_output)+(translateU(r_output,s_output)/2))
"""




"""
#bn inference
from pgmpy.models import BayesianModel

model3 = BayesianNetwork([('Rain', 'Wet_Grass'), ('Cloudy', 'Rain'), ('Cloudy', 'Sprinkler'), ('Sprinkler', 'Wet_Grass')])

from pgmpy.estimators import MaximumLikelihoodEstimator
mle = MaximumLikelihoodEstimator(model3, df)
cpd_rain_bn = mle.estimate_cpd('Rain')
cpd_cloudy_bn = mle.estimate_cpd('Cloudy')
cpd_sprinkler_bn = mle.estimate_cpd('Sprinkler')
cpd_wet_grass_bn = mle.estimate_cpd('Wet_Grass')

model3.add_cpds(cpd_cloudy_bn, cpd_wet_grass_bn, cpd_rain_bn, cpd_sprinkler_bn)
print(model3.check_model())

infer_bn_right = VariableElimination(model3)

q_bn_right = infer_bn_right.query(variables=['Rain'], evidence={'Wet_Grass': 1, "Cloudy": 1})

print("bn:",q_bn_right.get_value(Rain=0))
#bn inference


print("difference:",abs(q_bn_right.get_value(Rain=0)-translateD(r_output, s_output) + (translateU(r_output, s_output) / 2)))
"""



