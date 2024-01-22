import numpy as np
M1FLOW = np.round(np.array([3359+640,6007+1229,5349+1080,5563+1139,5299+1107]))
R3FLOW = np.round(np.array([480,1153,1129,1176,1095]))
M1A = [0.75,0.25]
V_ratio = [0.1,0.1,0.4,0.4]

EDGES = ['m2 m3 m4 m5 m6 m7 m8 m9',\
         'm2 m3 m4 m5 m6 m7 rout1 rout2',\
         'rin2 rin3 m7 m8 m9']


with open("fcd3.rou.xml", 'w') as routes:
    routes.write("""<routes>""" + '\n')
    routes.write('\n')
    routes.write(
        """<vType id="type0" color="255,105,180" length = "8.0" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
    routes.write(
        """<vType id="type1" color="255,190,180" length = "8.0" carFollowModel = "IDM" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
    routes.write(
        """<vType id="type2" color="22,255,255" length = "3.5" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
    routes.write(
        """<vType id="type3" color="22,55,255" length = "3.5" carFollowModel = "IDM" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
    routes.write('\n')
    for i in range(len(EDGES)):
        routes.write("""<route id=\"""" + str(i) + """\"""" + """ edges=\"""" + EDGES[i] + """\"/> """ + '\n')
    temp = 0
    for hours in range(len(M1FLOW)):
        m_in = np.random.poisson(lam=int(M1FLOW[hours,]))
        r3_in = np.random.poisson(lam=int(R3FLOW[hours,]))
        vNum = m_in + r3_in
        dtime = np.random.uniform(0 + 3600 * hours, 3600 + 3600 * hours, size=(int(vNum),))
        dtime.sort()
        for veh in range(int(vNum)):
            typev = np.random.choice([0, 1, 2, 3], p=V_ratio)
            vType = 'type' + str(typev)
            route = np.random.choice([0, 1, 2], p=[m_in * M1A[0] / vNum, m_in * M1A[1] / vNum, r3_in / vNum])
            routes.write("""<vehicle id=\"""" + str(temp + veh) + """\" depart=\"""" + str(
                round(dtime[veh], 2)) + """\" type=\"""" + str(vType) + """\" route=\"""" + str(
                route) + """\" departLane=\""""'random'"""\">""" + '\n')
            routes.write("""<param key = "has.ssm.device" value = "true"/>""" + '\n')
            # routes.write("""<param key = "device.ssm.measures" value = "TTC "/>""" + '\n')
            # routes.write("""<param key = "device.ssm.thresholds" value = "3.0 "/>""" + '\n')
            # routes.write("""<param key = "device.ssm.range" value = "50.0"/>""" + '\n')
            # routes.write("""<param key = "has.tripinfo.device" value="true"/>""" + '\n')
            routes.write("""</vehicle>""")
            routes.write('\n')
        temp += vNum
    routes.write("""</routes>""")