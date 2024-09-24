from sympy import symbols, Eq, solve, nsimplify, Rational

# Solving for \psi(r-3), \psi(r-2), ..., \psi(r+2) in terms of r and \psi(r)
u, v, x, y, z, a, r = symbols('u v x y z a r')
eq1 = Eq(v + a + z, 1/2)
eq2 = Eq(u + x + y, 1/2)
eq3 = Eq(3*u + 2*v + x - y - 2*z, r)
eq4 = Eq(9*u + 4*v + x + y + 4*z, r**2)
eq5 = Eq(27*u + 8*v + x - y - 8*z, r**3)
solution = solve([eq1, eq2, eq3, eq4, eq5], [u, v, x, y, z], rational=True)
print("solution to the system is: ", solution, end="\nValidation test:\n")

expr_list = [
    (solution[u] + solution[x] + solution[y], Rational(1, 2), "eq1"),
    (solution[v] + solution[z] + a, Rational(1, 2), "eq2"),
    (3*solution[u] + 2*solution[v] + solution[x] - solution[y] - 2*solution[z], r, "eq3"),
    (9*solution[u] + 4*solution[v] + solution[x] + solution[y] + 4*solution[z], r**2, "eq4"),
    (27*solution[u] + 8*solution[v] + solution[x] - solution[y] - 8*solution[z], r**3, "eq5")
]

def assert_equal_expr(expr, expected_expr, label):
    simplified_expr = nsimplify(expr)
    # print(simplified_expr)
    assert simplified_expr == expected_expr, f"Assertion failed: {label}, {simplified_expr}"
    print(f"{label}: Assertion passed. Simplified Expression = {simplified_expr}, Expected = {expected_expr}")
for i, j, string in expr_list:
    assert_equal_expr(i, j, string)
print()

expression_with_a = nsimplify(
    solution[u]**2 +
    solution[v]**2 +
    solution[x]**2 +
    solution[y]**2 +
    solution[z]**2 +
    a**2
).expand().collect(a)
print("sum of square condition is: ", expression_with_a, " = 67/128\n")

A = Rational(7)
B = (-Rational(7, 6) * r**3
    + Rational(11, 4) * r**2
    + Rational(11, 3) * r
    - Rational(61, 8))
C = (Rational(5, 72) * r**6
      - Rational(7, 24) * r**5
      - Rational(37, 288) * r**4
      + Rational(79, 48) * r**3
      - Rational(253, 288) * r**2
      - Rational(55, 24) * r
      + Rational(65, 32))

discriminant_rational = (B**2 - 4*A*C).simplify()
print(f"discriminant is {discriminant_rational}")
final_s1 = ((-B + discriminant_rational**0.5) / (2*A)).nsimplify()
final_s2 = ((-B - discriminant_rational**0.5) / (2*A)).nsimplify()
print("plug in r = 0, + sign = ", final_s1.subs({r: 0}).nsimplify())
print("plug in r = 0, - sign = ", final_s2.subs({r: 0}).nsimplify())
print("so we should take the + sign\n")
print(f"solution on 0-1 is:\n{final_s1}")

substitutions = {a: final_s1}
solutions_substituted = {var: sol.subs(substitutions) for var, sol in solution.items()}
print("Other piecewise function expressions:")
for var, sol in solutions_substituted.items():
    print(f"{var} = {sol}")

print("\nValidation at 0:", {key: expr.subs(r, 0) for key, expr in solutions_substituted.items()})
print("Validation at 1:", {key: expr.subs(r, 1) for key, expr in solutions_substituted.items()})
