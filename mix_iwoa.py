import time
import numpy as np

def mix_iwoa(search_space, objective_func, pop_size=15, max_iter=20, seed=42):
    """
    Mixed-Variable Improved Whale Optimization Algorithm (MixIWOA).
    
    Komponen Utama:
    1. Inisialisasi populasi menggunakan Peta Kaotik Logistik (u = 4.0).
    2. Faktor konvergensi non-linier: a = 2 * (1 - (t / max_iter)^2).
    3. Penanganan variabel campuran:
       - Kontinu (float) & Diskrit (int): Clipping batas + pembulatan untuk int.
       - Kategorikal: Transformasi Sigmoid + argmax one-hot decoding.

    Parameters
    ----------
    search_space : list of dict
        Definisi ruang pencarian hyperparameter. Contoh:
        [
            {'name': 'units', 'type': 'int', 'range': [32, 256]},
            {'name': 'lr', 'type': 'float', 'range': [0.0001, 0.01]},
            {'name': 'act', 'type': 'categorical', 'options': ['tanh', 'relu', 'sigmoid']},
            {'name': 'opt', 'type': 'categorical', 'options': ['Adam', 'RMSprop', 'SGD']}
        ]
    objective_func : callable
        Fungsi evaluasi yang menerima dictionary parameter hasil decode (`cand_params`)
        dan mengembalikan nilai float tunggal (fitness/loss yang ingin diminimalkan) 
        atau tuple `(fitness, eval_time)`.
    pop_size : int, default=15
        Jumlah agen populasi (paus).
    max_iter : int, default=20
        Jumlah iterasi maksimum.
    seed : int, default=42
        Random seed untuk reprodusibilitas eksperimen.

    Returns
    -------
    dict
        Dictionary berisi:
        - 'best_params': Kombinasi hyperparameter terbaik (dict).
        - 'best_fitness': Nilai fitness terbaik (float).
        - 'history': Riwayat nilai fitness terbaik di setiap iterasi (list).
        - 'total_time': Total waktu eksekusi keseluruhan dalam detik (float).
        - 'eval_time': Total waktu evaluasi fungsi objektif dalam detik (float).
        - 'overhead_time': Waktu komputasi murni algoritma MixIWOA dalam detik (float).
    """
    np.random.seed(seed)
    t_start = time.time()
    total_eval_time = 0.0

    # --- 1. Helper Internal (Enkapsulasi penuh dalam satu fungsi) ---
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def _logistic_chaotic_map(size, dimension):
        pop = np.zeros((size, dimension))
        u = 4.0
        for j in range(dimension):
            x_t = np.random.rand()
            # Hindari titik tetap (fixed points) pada peta logistik: 0, 0.25, 0.5, 0.75, 1.0
            while x_t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x_t = np.random.rand()
            for i in range(size):
                x_t = u * x_t * (1.0 - x_t)
                pop[i, j] = x_t
        return pop

    def _decode(vec):
        decoded = {}
        idx = 0
        for p in search_space:
            if p['type'] == 'categorical':
                n_opts = len(p['options'])
                segment = vec[idx : idx + n_opts]
                best_opt_idx = int(np.argmax(segment))
                decoded[p['name']] = p['options'][best_opt_idx]
                idx += n_opts
            elif p['type'] == 'int':
                decoded[p['name']] = int(round(vec[idx]))
                idx += 1
            else:  # float
                decoded[p['name']] = float(vec[idx])
                idx += 1
        return decoded

    def _evaluate(cand_params):
        nonlocal total_eval_time
        t0 = time.time()
        res = objective_func(cand_params)
        if isinstance(res, tuple):
            fit, t_ev = res[0], res[1]
        else:
            fit, t_ev = float(res), (time.time() - t0)
        total_eval_time += t_ev
        return fit

    # --- 2. Konstruksi Dimensi & Batas Ruang Pencarian ---
    dim = 0
    bounds_min, bounds_max = [], []
    for p in search_space:
        if p['type'] == 'categorical':
            n_opts = len(p['options'])
            dim += n_opts
            bounds_min.extend([0.0] * n_opts)
            bounds_max.extend([1.0] * n_opts)
        else:
            dim += 1
            bounds_min.append(float(p['range'][0]))
            bounds_max.append(float(p['range'][1]))

    bounds_min = np.array(bounds_min)
    bounds_max = np.array(bounds_max)

    # --- 3. Inisialisasi Populasi Kaotik ---
    chaotic_norm = _logistic_chaotic_map(pop_size, dim)
    pos = np.zeros((pop_size, dim + 1))  # Kolom terakhir menyimpan nilai fitness
    pos[:, :-1] = bounds_min + chaotic_norm * (bounds_max - bounds_min)

    for i in range(pop_size):
        cand = _decode(pos[i, :-1])
        pos[i, -1] = _evaluate(cand)

    leader = pos[np.argmin(pos[:, -1])].copy()
    history = [float(leader[-1])]

    # --- 4. Loop Utama Optimasi MixIWOA ---
    for it in range(max_iter):
        # Faktor peluruhan non-linier
        a = 2.0 * (1.0 - (it / max_iter) ** 2)

        for i in range(pop_size):
            r1, r2, p_rand = np.random.rand(), np.random.rand(), np.random.rand()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            new_vec = np.zeros(dim)

            for j in range(dim):
                if p_rand < 0.5:
                    if abs(A) >= 1.0:
                        # Fase Eksplorasi (Search for prey)
                        rand_idx = np.random.randint(0, pop_size)
                        X_rand = pos[rand_idx, :-1]
                        D = abs(C * X_rand[j] - pos[i, j])
                        new_vec[j] = X_rand[j] - A * D
                    else:
                        # Fase Eksploitasi (Shrinking encircling mechanism)
                        D = abs(C * leader[j] - pos[i, j])
                        new_vec[j] = leader[j] - A * D
                else:
                    # Fase Pembaruan posisi spiral (Spiral updating position)
                    D = abs(leader[j] - pos[i, j])
                    l = np.random.uniform(-1.0, 1.0)
                    new_vec[j] = D * np.exp(l) * np.cos(l * 2.0 * np.pi) + leader[j]

            # Boundary handling & Sigmoid transformation untuk dimensi kategorikal
            curr = 0
            for p in search_space:
                if p['type'] == 'categorical':
                    n_opts = len(p['options'])
                    new_vec[curr : curr + n_opts] = _sigmoid(new_vec[curr : curr + n_opts])
                    curr += n_opts
                else:
                    new_vec[curr] = np.clip(new_vec[curr], bounds_min[curr], bounds_max[curr])
                    curr += 1

            # Evaluasi kandidat baru
            pos[i, :-1] = new_vec
            cand = _decode(new_vec)
            pos[i, -1] = _evaluate(cand)

        # Update Global Best (Leader)
        current_best = pos[np.argmin(pos[:, -1])]
        if current_best[-1] < leader[-1]:
            leader = current_best.copy()

        history.append(float(leader[-1]))

    total_time = time.time() - t_start
    overhead_time = max(0.0, total_time - total_eval_time)

    return {
        'best_params': _decode(leader[:-1]),
        'best_fitness': float(leader[-1]),
        'history': history,
        'total_time': total_time,
        'eval_time': total_eval_time,
        'overhead_time': overhead_time
    }