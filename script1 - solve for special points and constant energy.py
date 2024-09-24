from sympy import symbols, Eq, solve

## Solving for \psi(-2), ..., \psi(2)
v, x, y, z, a = symbols('v x y z a')
eq1 = Eq(v + a + z, 1/2)
eq2 = Eq(x + y, 1/2)
eq3 = Eq(-2*v - x + y + 2*z, 0)
eq4 = Eq(4*v + x + y + 4*z, 0)
eq5 = Eq(-8*v - x + y + 8*z, 0)
solution = solve([eq1, eq2, eq3, eq4, eq5], [v, x, y, z, a], rational=True)
print("solution to the system is: ", solution)

sum_constant = 0
for val, sol in solution.items():
    sum_constant += sol**2
print("Constant C =", sum_constant)