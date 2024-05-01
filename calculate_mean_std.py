f1_scores = [0.84, 0.74, 0.82]
# accs = [0.77, 0.81, 0.76]

f1_mean = sum(f1_scores) / len(f1_scores)
f1_std = (sum([(x - f1_mean) ** 2 for x in f1_scores]) / len(f1_scores)) ** 0.5

# acc_mean = sum(accs) / len(accs)
# acc_std = (sum([(x - acc_mean) ** 2 for x in accs]) / len(accs)) ** 0.5

print(f"F1: {f1_mean} ± {f1_std}")
# print(f"Accuracy: {acc_mean} ± {acc_std}")