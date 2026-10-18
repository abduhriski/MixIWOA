from mix_iwoa import mix_iwoa

# 1. Definisikan ruang pencarian
SEARCH_SPACE = [
    {'name': 'units', 'type': 'int', 'range': [32, 256]},
    {'name': 'lr', 'type': 'float', 'range': [0.0001, 0.01]},
    {'name': 'act', 'type': 'categorical', 'options': ['tanh', 'relu', 'sigmoid']},
    {'name': 'opt', 'type': 'categorical', 'options': ['Adam', 'RMSprop', 'SGD']}
]

# 2. Bungkus fungsi evaluasi LSTM Anda dengan lambda
obj_fn = lambda params: train_evaluate_lstm(
    params, data['X_train'], data['y_train'], data['X_val'], data['y_val']
)

# 3. Jalankan optimasi
result = mix_iwoa(
    search_space=SEARCH_SPACE,
    objective_func=obj_fn,
    pop_size=15,
    max_iter=20,
    seed=SEED
)

print("Hyperparameter Terbaik:", result['best_params'])
print("Validation RMSE Terbaik:", result['best_fitness'])