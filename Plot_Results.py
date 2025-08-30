import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.gridspec as gridspec




FirstStageIsp = np.linspace(0.85,1.15, 13)-1
Falcon9_LEO_EXP_FirstStageIsp=[18798.347, 19662.062, 20540.639, 21439.065, 22358.155, 23297.030, 24258.750, 25241.651, 26248.189, 27278.628, 28327.543, 29397.265, 30489.063]
Falcon9_LEO_ASDS_FirstStageIsp = [16666.9,17394.6,18133.5, 18768.0, 19592.6,20357.6,21136.6,21961.4,22821.0,23711.9,24630.0, 25567.9,26526.2]
Falcon9_LEO_RTLS_FirstStageIsp = [13562.1,14130.2,14730.3,15300.3,15921.4, 16556.0,17208.8,17838.7,18565.8,19187.4, 19964.1, 20604.2, 21433.5]
Falcon9_MEO_EXP_FirstStageIsp=[6598.093, 6998.104, 7407.646, 7826.264, 8254.697, 8694.045, 9142.798, 9603.462, 10075.874, 10559.185, 11054.675, 11562.171, 12085.387]
Falcon9_MEO_ASDS_FirstStageIsp=[5494.155, 5820.877, 6139.110, 6466.210, 6803.847, 7164.032, 7540.683, 7932.160, 8336.508, 8750.859, 9176.348, 9614.795, 10066.929]
Falcon9_MEO_RTLS_FirstStageIsp=[4052.452, 4214.697, 4467.885, 4722.328, 4986.838, 5252.924, 5552.885, 5841.886, 6135.371, 6423.243, 6743.008, 7033.621, 7368.914]
Falcon9_TLI_EXP_FirstStageIsp=[4703.055, 5028.978, 5362.283, 5702.469, 6051.485, 6409.238, 6776.344, 7153.786, 7540.376, 7937.467, 8343.937, 8760.245, 9188.197]
Falcon9_TLI_ASDS_FirstStageIsp=[3771.456, 4051.484, 4260.443, 4576.815, 4850.988, 5142.696, 5447.867, 5764.729, 6092.100, 6430.224, 6776.814, 7133.553, 7501.835]
Falcon9_TLI_RTLS_FirstStageIsp=[2529.855, 2732.117, 2925.401, 3124.672, 3345.187, 3571.376, 3798.262, 4031.591, 4262.553, 4509.325, 4728.074, 5002.890, 5266.441]

SecondStageIsp = np.linspace(0.85,1.15, 13)-1
Falcon9_LEO_EXP_SecondStageIsp=[17810.359, 18898.267, 19980.920, 21057.286, 22129.422, 23196.729, 24258.750, 25315.243, 26365.716, 27409.893, 28446.840, 29476.133, 30497.963]
Falcon9_LEO_ASDS_SecondStageIsp=[14918.338, 15959.559, 16999.389, 18038.819, 19074.957, 20107.514, 21136.592, 22162.786, 23184.990, 24204.207, 25218.839, 26060.686, 27133.612]
Falcon9_LEO_RTLS_SecondStageIsp=[11341.156, 12309.069, 13282.877, 14242.298, 15157.905, 16225.299, 17208.841, 18173.213, 19173.251, 20152.442, 21083.143, 22106.618, 23081.064]
Falcon9_MEO_EXP_SecondStageIsp=[5019.773, 5684.964, 6360.384, 7044.875, 7737.357, 8436.949, 9142.798, 9854.211, 10571.153, 11293.102, 12019.567, 12749.896, 13483.687]
Falcon9_MEO_ASDS_SecondStageIsp=[3717.509, 4327.744, 4950.182, 5583.724, 6227.295, 6879.919, 7540.683, 8208.756, 8883.369, 9563.814, 10248.971, 10939.271, 11634.349]
Falcon9_MEO_RTLS_SecondStageIsp=[2137.677, 2650.914, 3217.788, 3788.149, 4365.014, 4946.606, 5552.885, 6162.100, 6772.736, 7406.821, 8040.898, 8681.869, 9320.555]
Falcon9_TLI_EXP_SecondStageIsp=[3174.546, 3748.484, 4334.108, 4930.366, 5536.438, 6152.024, 6776.344, 7408.689, 8048.324, 8694.575, 9346.885, 10004.696, 10667.416]
Falcon9_TLI_ASDS_SecondStageIsp=[2133.413, 2656.269, 3192.246, 3740.329, 4299.572, 4869.076, 5447.867, 6035.195, 6630.958, 7234.657, 7845.143, 8463.353, 9087.234]
Falcon9_TLI_RTLS_SecondStageIsp=[863.193, 1322.151, 1772.391, 2274.505, 2770.584, 3264.552, 3798.262, 4328.271, 4868.078, 5416.940, 5974.224, 6532.235, 7072.020]

Starship_LEO_EXP_FirstStageIsp=[208560.879, 217050.668, 225555.004, 234193.650, 242895.577, 251643.242, 260550.507, 269580.615, 278771.442, 288136.964, 288136.964, 307297.481, 317086.937]
Starship_LEO_ASDS_FirstStageIsp=[179697.728, 183548.596, 194908.020, 202481.471, 210840.167, 218999.890, 225645.063, 235379.572, 243718.239, 252101.061, 260601.656, 269274.086, 279028.183]
Starship_LEO_RTLS_FirstStageIsp=[161983.606, 168739.061, 175579.271, 182394.853, 189295.688, 196314.869, 202874.769, 210117.240, 216966.957, 225336.104, 231959.665, 239520.233, 248608.529]
Starship_MEO_EXP_FirstStageIsp=[58626.182, 62968.606, 67396.246, 71848.836, 76400.679, 80973.914, 85652.152, 90357.375, 95132.659, 99984.644, 104871.905, 109871.837, 114967.118]
Starship_MEO_ASDS_FirstStageIsp=[44728.293, 48439.378, 50186.016, 55903.931, 65894.028, 64724.598, 67149.379, 71363.972, 75674.774, 80019.553, 83910.787, 97118.688, 91081.161]
Starship_MEO_RTLS_FirstStageIsp=[32730.683, 36024.431, 38869.376, 42590.581, 45947.571, 49188.234, 52519.570, 56066.893, 59601.616, 52310.278, 66891.563, 70589.355, 66938.675]
Starship_TLI_EXP_FirstStageIsp=[35522.241, 39045.021, 42632.961, 46244.680, 49924.476, 49924.476, 57402.658, 61242.838, 65170.649, 69182.755, 73275.409, 77453.040, 81710.045]
Starship_TLI_ASDS_FirstStageIsp=[24079.834, 31587.438, 34778.003, 38039.430, 41385.670, 38913.060, 42300.371, 45713.792, 49188.606, 59249.289, 56249.244, 66742.481, 70484.858]
Starship_TLI_RTLS_FirstStageIsp=[14100.565, 16714.852, 19357.639, 22044.402, 24704.071, 27360.602, 30104.624, 32917.090, 26693.972, 38710.647, 41653.422, 44621.845, 47647.240]
Starship_LEO_EXP_SecondStageIsp=[175370.808, 190121.233, 204674.749, 219040.843, 233102.126, 246933.905, 260550.507, 274060.678, 287446.373, 300702.600, 313828.916, 326817.799, 339663.247]
Starship_LEO_ASDS_SecondStageIsp=[143577.536, 156222.993, 172128.711, 186164.149, 200075.595, 215666.932, 227119.710, 254251.273, 253273.064, 266087.003, 278782.366, 291358.023, 303808.436]
Starship_LEO_RTLS_SecondStageIsp=[119443.500, 133588.078, 147558.233, 161864.698, 175796.093, 189571.766, 202874.769, 216664.516, 229835.360, 242267.874, 255535.632, 268147.251, 279967.985]
Starship_MEO_EXP_SecondStageIsp=[27892.494, 37302.194, 46827.763, 56435.658, 66123.520, 75843.931, 85652.152, 95420.509, 105210.589, 114967.245, 124777.188, 134631.096, 144520.660]
Starship_MEO_ASDS_SecondStageIsp=[11897.747, 27568.712, 30593.421, 39350.727, 48527.016, 57798.490, 67149.379, 76522.125, 85926.839, 95363.990, 104558.311, 114160.846, 123543.091]
Starship_MEO_RTLS_SecondStageIsp=[406.366, 8674.101, 17149.148, 25805.512, 30211.051, 43444.325, 52519.570, 61687.520, 70883.122, 80095.551, 89349.386, 98654.950, 107913.817]
Starship_TLI_EXP_SecondStageIsp=[7306.455, 15372.688, 23575.508, 31893.428, 40318.251, 48834.134, 57402.658, 66086.265, 74873.563, 83753.553, 92711.856, 101741.221, 110833.861]
Starship_TLI_ASDS_SecondStageIsp=[-121.519, 2583.678, 10476.753, 18091.277, 26330.971, 34441.468, 42300.371, 56733.880, 58832.713, 67183.062, 75628.149, 84158.302, 99912.673]
Starship_TLI_RTLS_SecondStageIsp=[-14816.292, -7765.519, -507.783, 6928.172, 14516.150, 22246.974, 30104.624, 38075.048, 46016.904, 54213.590, 50999.355, 70591.238, 78877.200]




color_vec = ['r','b','g','b','k','y','c','m','olive']
plt.figure()
plt.subplot(1,2,1)
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_LEO_EXP_FirstStageIsp), '-s', color=color_vec[0], label='LEO_EXP')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_LEO_ASDS_FirstStageIsp), '-s', color=color_vec[1], label='LEO_ASDS')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_LEO_RTLS_FirstStageIsp), '-s', color=color_vec[2], label='LEO_RTLS')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_MEO_EXP_FirstStageIsp), '-o', color=color_vec[0], label='MEO_EXP')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_MEO_ASDS_FirstStageIsp), '-o', color=color_vec[1], label='MEO_ASDS')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_MEO_RTLS_FirstStageIsp), '-o', color=color_vec[2], label='MEO_RTLS')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_TLI_EXP_FirstStageIsp), '-^', color=color_vec[0], label='TLI_EXP')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_TLI_ASDS_FirstStageIsp), '-^', color=color_vec[1], label='TLI_ASDS')
plt.plot(FirstStageIsp*100, 0.001*np.array(Falcon9_TLI_RTLS_FirstStageIsp), '-^', color=color_vec[2], label='TLI_RTLS')

plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_LEO_EXP_SecondStageIsp), '--s', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_LEO_ASDS_SecondStageIsp), '--s', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_LEO_RTLS_SecondStageIsp), '--s', color=color_vec[2])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_MEO_EXP_SecondStageIsp), '--o', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_MEO_ASDS_SecondStageIsp), '--o', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_MEO_RTLS_SecondStageIsp), '--o', color=color_vec[2])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_TLI_EXP_SecondStageIsp), '--^', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_TLI_ASDS_SecondStageIsp), '--^', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Falcon9_TLI_RTLS_SecondStageIsp), '--^', color=color_vec[2])
plt.xlabel('First and Second Stage Isp Change [%]')
plt.ylabel('Payload Capacity [Tons]')
plt.title('Falcon 9 Payload Capacity Sensativity to Engine ISP')
plt.grid('on')
plt.legend(ncol = 2)

plt.subplot(1,2,2)
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_LEO_EXP_FirstStageIsp), '-s', color=color_vec[0])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_LEO_ASDS_FirstStageIsp), '-s', color=color_vec[1])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_LEO_RTLS_FirstStageIsp), '-s', color=color_vec[2])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_MEO_EXP_FirstStageIsp), '-o', color=color_vec[0])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_MEO_ASDS_FirstStageIsp), '-o', color=color_vec[1])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_MEO_RTLS_FirstStageIsp), '-o', color=color_vec[2])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_TLI_EXP_FirstStageIsp), '-^', color=color_vec[0])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_TLI_ASDS_FirstStageIsp), '-^', color=color_vec[1])
plt.plot(FirstStageIsp*100, 0.001*np.array(Starship_TLI_RTLS_FirstStageIsp), '-^', color=color_vec[2])

plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_LEO_EXP_SecondStageIsp), '--s', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_LEO_ASDS_SecondStageIsp), '--s', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_LEO_RTLS_SecondStageIsp), '--s', color=color_vec[2])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_MEO_EXP_SecondStageIsp), '--o', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_MEO_ASDS_SecondStageIsp), '--o', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_MEO_RTLS_SecondStageIsp), '--o', color=color_vec[2])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_TLI_EXP_SecondStageIsp), '--^', color=color_vec[0])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_TLI_ASDS_SecondStageIsp), '--^', color=color_vec[1])
plt.plot(SecondStageIsp*100, 0.001*np.array(Starship_TLI_RTLS_SecondStageIsp), '--^', color=color_vec[2])
plt.xlabel('First and Second Stage Isp Change [%]')
plt.ylabel('Payload Capacity [Tons]')
plt.title('Starship Payload Capacity Sensativity to Engine ISP')
plt.grid('on')

plt.show()


























