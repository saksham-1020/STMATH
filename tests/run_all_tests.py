import stmath as am

def safe_run(label, func, *args, **kwargs):
    try:
        res = func(*args, **kwargs)
        print(label, res)
    except Exception as e:
        print(label, f"ERROR: {e}")

print("=== CORE ===")
safe_run(">> 1:", am.add, 10, 5)
safe_run(">> 2:", am.sub, 10, 5)
safe_run(">> 3:", am.mul, 10, 5)
safe_run(">> 4:", am.div, 10, 5)
safe_run(">> 5:", am.square, 4)
safe_run(">> 6:", am.cube, 3)
safe_run(">> 7:", am.sqrt, 16)
safe_run(">> 8:", am.power, 2, 3)
safe_run(">> 9:", am.percent, 50, 200)
safe_run(">> 10:", am.percent_change, 100, 120)

print("\n=== SCIENTIFIC ===")
safe_run(">> 11:", am.sin, am.pi/2)
safe_run(">> 12:", am.cos, 0)
safe_run(">> 13:", am.tan, am.pi/4)
safe_run(">> 14:", am.log10, 100)
safe_run(">> 15:", am.ln, am.e)
safe_run(">> 16:", am.exp, 1)
safe_run(">> 17:", am.factorial, 5)
safe_run(">> 18:", am.gcd, 12, 18)
safe_run(">> 19:", am.lcm, 12, 18)
safe_run(">> 20:", am.deg2rad, 180)
safe_run(">> 21:", am.rad2deg, am.pi)

print("\n=== STATISTICS ===")
safe_run(">> 22:", am.mean, [10,20,30])
safe_run(">> 23:", am.median, [10,20,30])
safe_run(">> 24:", am.mode, [10,20,20,30])
safe_run(">> 25:", am.variance, [10,20,30])
safe_run(">> 26:", am.std, [10,20,30])
safe_run(">> 27:", am.data_range, [10,20,30])
safe_run(">> 28:", am.iqr, [1,2,3,4,5,6,7,8])
safe_run(">> 29:", am.z_score, 70, 60, 5)

print("\n=== PROBABILITY & DISTRIBUTIONS ===")
safe_run(">> 30:", am.nCr, 5, 2)
safe_run(">> 31:", am.nPr, 5, 2)
safe_run(">> 32:", am.bayes, 0.5, 0.4, 0.7)
safe_run(">> 33:", am.expected_value, [1,2,3], [0.2,0.3,0.5])
safe_run(">> 34:", am.normal_pdf, 0, 0, 1)
safe_run(">> 35:", am.normal_cdf, 0, 0, 1)
safe_run(">> 36:", am.bernoulli_pmf, 1, 0.6)
safe_run(">> 37:", am.binomial_pmf, 2, 5, 0.5)
safe_run(">> 38:", am.poisson_pmf, 3, 2)
safe_run(">> 39:", am.exponential_pdf, 2, 1)
safe_run(">> 40:", am.uniform_pdf, 2, 0, 5)
safe_run(">> 41:", am.t_pdf, 0, 10)
safe_run(">> 42:", am.chi_square_pdf, 2, 4)

print("\n=== MACHINE LEARNING METRICS ===")
safe_run(">> 43:", am.mse, [1,2,3], [1,2,4])
safe_run(">> 44:", am.rmse, [1,2,3], [1,2,4])
safe_run(">> 45:", am.mae, [1,2,3], [1,2,4])
safe_run(">> 46:", am.accuracy, [1,0,1], [1,0,0])
safe_run(">> 47:", am.precision, [1,0,1,0], [1,1,1,0])
safe_run(">> 48:", am.recall, [1,0,1,0], [1,1,1,0])
safe_run(">> 49:", am.f1_score, [1,0,1,0], [1,1,1,0])
safe_run(">> 50:", am.r2_score, [1,2,3], [1,2,4])

print("\n=== DEEP LEARNING ===")
safe_run(">> 51:", am.sigmoid, 0)
safe_run(">> 52:", am.relu, -3)
safe_run(">> 53:", am.tanh, 1)
safe_run(">> 54:", am.softmax, [1,2,3])
safe_run(">> 55:", am.entropy, [0.5,0.5])
safe_run(">> 56:", am.kl_divergence, [0.5,0.5], [0.9,0.1])
safe_run(">> 57:", am.binary_cross_entropy, [1,0], [0.9,0.2])

print("\n=== NLP ===")
safe_run(">> 58:", am.term_frequency, "machine", ["machine","learning","machine"])
safe_run(">> 59:", am.inverse_document_frequency, "machine", [["machine","learning"],["deep","learning"]])
safe_run(">> 60:", am.tfidf, "machine", ["machine","learning"], [["machine","learning"],["deep","learning"]])
safe_run(">> 61:", am.cosine_similarity, [1,0,1], [0,1,1])
safe_run(">> 62:", am.perplexity, [0.25,0.25,0.25,0.25])

print("\n=== TIME SERIES ===")
safe_run(">> 63:", am.sma, [10,20,30,40,50], 3)
safe_run(">> 64:", am.ema, [10,20,30,40,50], 0.5)

print("\n=== FINANCE ===")
safe_run(">> 65:", am.simple_interest, 1000, 10, 2)
safe_run(">> 66:", am.compound_interest, 1000, 10, 2)
safe_run(">> 67:", am.loan_emi, 500000, 7.5, 240)

print("\n=== CRYPTO ===")
safe_run(">> 68:", am.sha256, "hello")
safe_run(">> 69:", am.gas_fee, 21000, 50, 2000)  # gas_used, gwei, eth_price in USD


print("\n=== QUANTUM ===")
safe_run(">> 70:", am.hadamard, [1,0])
safe_run(">> 71:", am.pauli_x, [1,0])


print("\n=== APTITUDE ===")
safe_run(">> 72:", am.profit_percent, 100, 120)
safe_run(">> 73:", am.loss_percent, 100, 80)
safe_run(">> 74:", am.avg_speed, 60, 1, 40, 1)

print("\n=== ALGEBRA ===")
safe_run(">> 75:", am.solve_linear, 2, 4)
safe_run(">> 76:", am.quadratic_roots, 1, -3, 2)

print("\n=== GEN-AI HELPERS ===")
safe_run(">> 77:", am.logits_to_prob, [2.0, 1.0, 0.1])
safe_run(">> 78:", am.softmax_temperature, [2.0, 1.0, 0.1], 0.7)
safe_run(">> 79:", am.attention_scores, [[1,0],[0,1]], [[1,0],[0,1]], [[1],[2]])

print("\n=== OPTIMIZATION ===")
safe_run(">> 80:", am.sgd_update, [0.5, -0.3], [0.1, -0.2], 0.01)
safe_run(">> 81:", am.adam_update, [0.5, -0.3], [0.1, -0.2], [0,0], [0,0], 0.001, 0.9, 0.999, 1e-8)
safe_run(">> 82:", am.rmsprop_update, [0.5, -0.3], [0.1, -0.2], [0,0], 0.001, 0.9, 1e-8)
safe_run(">> 83:", am.lr_step_decay, 0.1, 10, 0.5)
safe_run(">> 84:", am.lr_cosine_anneal, 0.1, 0.001, 50)
safe_run(">> 85:", am.momentum_update, [0.5, -0.3], [0.1, -0.2], [0,0], 0.01, 0.9)

print("\n=== GRAPH ===")
graph_unweighted = {0:[1,2],1:[2],2:[3],3:[]}
safe_run(">> 86:", am.bfs_distance, {0:[1,2],1:[2],2:[3],3:[]}, 0)


graph_weighted = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}


safe_run(">> 87:", am.dijkstra_shortest_path,
         {0:{1:4,2:1}, 1:{3:1}, 2:{1:2,3:5}, 3:{}}, 0)




print("\n=== VISION ===")
safe_run(">> 88:", am.conv2d_output_shape, (64,64), (3,3), 1, 0, 1)
safe_run(">> 89:", am.maxpool_output_shape, (64,64), 2, 2, 0)
safe_run(">> 90:", am.iou, (0,0,10,10), (5,5,15,15))
safe_run(">> 91:", am.nms, [(0,0,10,10,0.9),(1,1,9,9,0.8),(20,20,30,30,0.7)])

print("\n=== NUMBER THEORY ===")
safe_run(">> 92:", am.is_prime, 29)
safe_run(">> 93:", am.prime_factors, 84)
safe_run(">> 94:", am.totient, 9)
safe_run(">> 95:", am.mod_inverse, 3, 11)
safe_run(">> 96:", am.modular_pow, 2, 10, 1000)
safe_run(">> 97:", am.fibonacci, 7)
safe_run(">> 98:", am.pell_number, 5)
safe_run(">> 99:", am.catalan_number, 4)
safe_run(">> 100:", am.divisor_count, 28)
safe_run(">> 101:", am.divisor_sum, 28)

print("\n=== MATH EXT (spot check from math module) ===")

for i, (name, args) in enumerate([
    ( "sqrt", (144,) ),
    ( "log2", (8,) ),
    ( "cosh", (0.0,) ),
], start=102):
    if hasattr(am, name):
        safe_run(f">> {i}:", getattr(am, name), *args)
    else:
        print(f">> {i}:", f"SKIP: {name} not exposed")

print("\n=== DONE ===")
