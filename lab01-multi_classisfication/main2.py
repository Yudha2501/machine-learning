# TRAINING DI ESP32 — MicroPython tanpa numpy
TRAIN = [
    (800, 20, 122), (900, 50, 170), (1000, 30, 153),
    (1100, 70, 205), (1200, 40, 182), (1300, 80, 236),
    (1400, 25, 190), (1500, 60, 236), (1600, 45, 225),
    (1700, 85, 276), (1800, 35, 239), (1900, 75, 284)
]

TEST = [
    (950, 40, 164), (1250, 65, 213), (1450, 35, 203),
    (1750, 55, 258), (1850, 90, 298)
]


def solve_linear(a, b):
    # Gauss-Jordan + partial pivoting untuk matriks kecil
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-10:
            raise ValueError("Fitur tidak independen / matriks singular")
        scale = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= scale
        for row in range(n):
            if row != col:
                factor = m[row][col]
                for j in range(col, n + 1):
                    m[row][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def fit_multiple(data):
    # x = [1, rpm/1000, load/100]
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for rpm, load, power in data:
        x = [1.0, rpm / 1000.0, load / 100.0]
        for i in range(3):
            xty[i] += x[i] * power
            for j in range(3):
                xtx[i][j] += x[i] * x[j]
    return solve_linear(xtx, xty)


def predict_multiple(coef, rpm, load):
    return coef[0] + coef[1] * (rpm / 1000.0) + coef[2] * (load / 100.0)


def mae(data, predict_fn):
    total = 0.0
    for rpm, load, actual in data:
        total += abs(actual - predict_fn(rpm, load))
    return total / len(data)


coef = fit_multiple(TRAIN)
print("Coefficients for scaled features:", coef)
print("Test predictions:")
for rpm, load, actual in TEST:
    pred = predict_multiple(coef, rpm, load)
    print("RPM", rpm, "Load", load, "Actual", actual,
          "Pred", round(pred, 2))

print("Test MAE (W):", round(mae(
    TEST, lambda rpm, load: predict_multiple(coef, rpm, load)
), 4))

# Simple linear regression: daya = a + b * (rpm/1000)
def fit_rpm_only(data):
    n = len(data)
    xs = [row[0] / 1000.0 for row in data]
    ys = [row[2] for row in data]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Nilai RPM tidak bervariasi")
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return intercept, slope


simple_coef = fit_rpm_only(TRAIN)


def predict_simple(rpm, load):
    # load sengaja tidak digunakan oleh model satu fitur
    return simple_coef[0] + simple_coef[1] * (rpm / 1000.0)


simple_mae = mae(TEST, predict_simple)
multiple_mae = mae(TEST, lambda rpm, load:
                   predict_multiple(coef, rpm, load))
print("MAE RPM only (W):", round(simple_mae, 4))
print("MAE RPM + load (W):", round(multiple_mae, 4))
train_outlier = TRAIN + [(1400, 50, 500)]
coef_outlier = fit_multiple(train_outlier)
print("MAE test sebelum outlier:", round(mae(
    TEST, lambda r, l: predict_multiple(coef, r, l)), 4))
print("MAE test setelah outlier:", round(mae(
    TEST, lambda r, l: predict_multiple(coef_outlier, r, l)), 4))

while True:
    try:
        rpm = float(input("RPM (800-1900, atau Ctrl+C): "))
        load = float(input("Load % (0-100): "))
        if not (800 <= rpm <= 1900 and 0 <= load <= 100):
            print("Input di luar rentang yang disarankan.")
            continue
        power = predict_multiple(coef, rpm, load)
        print("Predicted power:", round(power, 2), "W")
        print("--------------------------")
    except ValueError:
        print("Masukkan angka yang valid.")
    except KeyboardInterrupt:
        print("Program dihentikan.")
        break