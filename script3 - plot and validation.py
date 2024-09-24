import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# load the piecewise expressions
r = sp.symbols('r')
sqrt_dis = sp.sqrt(sp.Rational(-7, 12)*r**6 + sp.Rational(7, 4)*r**5 + sp.Rational(125, 48)*r**4 - sp.Rational(65, 8)*r**3 - sp.Rational(187, 48)*r**2 + sp.Rational(33, 4)*r + sp.Rational(81, 64))
u = sp.Rational(1, 24)*r**3 + sp.Rational(3, 112)*r**2 - sp.Rational(11, 84)*r + sqrt_dis/sp.Rational(28) - sp.Rational(9, 224)
v = -sp.Rational(1, 24)*r**3 + sp.Rational(5, 112)*r**2 + sp.Rational(13, 42)*r - 3*sqrt_dis/28 + sp.Rational(13, 224)
x = -sp.Rational(1, 12)*r**3 + sp.Rational(3, 56)*r**2 + sp.Rational(17, 42)*r + sqrt_dis/14 + sp.Rational(19, 112)
a = sp.Rational(1, 12)*r**3 - sp.Rational(11, 56)*r**2 - sp.Rational(11, 42)*r + sqrt_dis/14 + sp.Rational(61, 112)
y = sp.Rational(1, 24)*r**3 - sp.Rational(9, 112)*r**2 - sp.Rational(23, 84)*r - 3*sqrt_dis/28 + sp.Rational(83, 224)
z = -sp.Rational(1, 24)*r**3 + sp.Rational(17, 112)*r**2 - sp.Rational(1, 21)*r + sqrt_dis/28 - sp.Rational(23, 224)

# transform the sympy expressions to numpy functions
u_func = sp.lambdify(r, u, 'numpy')
v_func = sp.lambdify(r, v, 'numpy')
x_func = sp.lambdify(r, x, 'numpy')
a_func = sp.lambdify(r, a, 'numpy')
y_func = sp.lambdify(r, y, 'numpy')
z_func = sp.lambdify(r, z, 'numpy')

# auto diff
u_prime = sp.diff(u, r)
v_prime = sp.diff(v, r)
x_prime = sp.diff(x, r)
a_prime = sp.diff(a, r)
y_prime = sp.diff(y, r)
z_prime = sp.diff(z, r)

u_prime_func = sp.lambdify(r, u_prime, 'numpy')
v_prime_func = sp.lambdify(r, v_prime, 'numpy')
x_prime_func = sp.lambdify(r, x_prime, 'numpy')
a_prime_func = sp.lambdify(r, a_prime, 'numpy')
y_prime_func = sp.lambdify(r, y_prime, 'numpy')
z_prime_func = sp.lambdify(r, z_prime, 'numpy')

def piecewise_function(value):
    if value <= -3 or value >= 3:
        return 0
    if -3 < value < -2:
        return u_func(value + 3)
    elif -2 <= value < -1:
        return v_func(value + 2)
    elif -1 <= value < 0:
        return x_func(value + 1)
    elif 0 <= value < 1:
        return a_func(value)
    elif 1 <= value < 2:
        return y_func(value - 1)
    elif 2 <= value < 3:
        return z_func(value - 2)

def piecewise_derivative(value):
    if value <= -3 or value >= 3:
        return 0
    if -3 < value < -2:
        return u_prime_func(value + 3)
    elif -2 <= value < -1:
        return v_prime_func(value + 2)
    elif -1 <= value < 0:
        return x_prime_func(value + 1)
    elif 0 <= value < 1:
        return a_prime_func(value)
    elif 1 <= value < 2:
        return y_prime_func(value - 1)
    elif 2 <= value < 3:
        return z_prime_func(value - 2)

def vectorized_piecewise_function(value):
    value = value.astype(np.complex128)
    return np.where(
        (value <= -3) | (value >= 3), 0,
        np.where(
            (-3 < value) & (value < -2), u_func(value + 3),
            np.where(
                (-2 <= value) & (value < -1), v_func(value + 2),
                np.where(
                    (-1 <= value) & (value < 0), x_func(value + 1),
                    np.where(
                        (0 <= value) & (value < 1), a_func(value),
                        np.where(
                            (1 <= value) & (value < 2), y_func(value - 1),
                            z_func(value - 2)
                        )
                    )
                )
            )
        )
    )

def vectorized_piecewise_derivative(value):
    value = value.astype(np.complex128)
    return np.where(
        (value <= -3) | (value >= 3), 0,
        np.where(
            (-3 < value) & (value < -2), u_prime_func(value + 3),
            np.where(
                (-2 <= value) & (value < -1), v_prime_func(value + 2),
                np.where(
                    (-1 <= value) & (value < 0), x_prime_func(value + 1),
                    np.where(
                        (0 <= value) & (value < 1), a_prime_func(value),
                        np.where(
                            (1 <= value) & (value < 2), y_prime_func(value - 1),
                            z_prime_func(value - 2)
                        )
                    )
                )
            )
        )
    )

# r_vals = np.linspace(0, 1, 20)
# plt.plot(r_vals - 4, vectorized_piecewise_function(r_vals - 4), label='psi on [-4, -3)')
# plt.plot(r_vals - 3, u_func(r_vals), label='u on [-3, -2)')
# plt.plot(r_vals - 2, v_func(r_vals), label='v on [-2, -1)')
# plt.plot(r_vals - 1, x_func(r_vals), label='x on [-1,  0)')
# plt.plot(r_vals,     a_func(r_vals), label='a on [0,   1)')
# plt.plot(r_vals + 1, y_func(r_vals), label='y on [1,   2)')
# plt.plot(r_vals + 2, z_func(r_vals), label='z on [2,   3)')
# plt.plot(r_vals + 3, vectorized_piecewise_function(r_vals + 3), label='psi on [3, 4)')
# plt.title('Piecewise Function Visualization')
# plt.xlabel('r')
# plt.ylabel('f(r)')
# plt.legend()
# plt.grid(True)
# plt.show()

r_vals = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))
plt.plot(r_vals, vectorized_piecewise_function(r_vals), label='six-point delta function', color='blue')
plt.plot(r_vals, vectorized_piecewise_derivative(r_vals), label='Derivative of six-point delta function', color='red', linestyle='dashed')
plt.title('Six-point delta function and its derivative')
plt.xlabel('r')
plt.ylabel('f(r) / f\'(r)')
plt.legend()
plt.grid(True)
plt.xlim([-4, 4])
plt.show()

sum_even = (v+a+z).simplify()
sum_odd = (u+x+y).simplify()
print(f"even sum = {sum_even}")
print(f"odd sum = {sum_odd}")

def test_conditions(r_value, order=1):
    total_sum = 0
    for i in range(-100, 100):
        phi_val = piecewise_function(r_value - i)
        total_sum += (r_value - i)**order * phi_val
    return total_sum

def test_phi_square_sum(r_value):
    sum_phi_square = 0
    for i in range(-100, 100):
        phi_val = piecewise_function(r_value - i)
        sum_phi_square += phi_val**2
    return sum_phi_square

def run_all_tests():
    r_test_values = np.random.uniform(5, -5, 1000)
    target_phi_square = 67 / 128
    tolerance = 1e-6
    valid_count = 0
    invalid_count = 0

    results = []
    for r_val in r_test_values:
        sum_1 = test_conditions(r_val, 1)  # test ∑(r-i)φ(r-i) = 0
        sum_2 = test_conditions(r_val, 2)  # test ∑(r-i)^2 φ(r-i) = 0
        sum_3 = test_conditions(r_val, 3)  # test ∑(r-i)^3 φ(r-i) = 0
        phi_square = test_phi_square_sum(r_val)  # test ∑(φ(r-i))^2 = C
        is_valid = (abs(sum_1) < tolerance and abs(sum_2) < tolerance and
                    abs(sum_3) < tolerance and abs(phi_square - target_phi_square) < tolerance)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
    return valid_count, invalid_count

valid_count, invalid_count = run_all_tests()
print(f"Of the 10k random test points, {valid_count} are valid and {invalid_count} are invalid.")