from pybfm.irr import IRR

# define annual cash flows
multiple_irr = IRR(
    [0, 1, 2],  # years [this year, first year, second year]
    [-3000, 15000, -13000],  # cash flows
    [None, None, None],  # kind of cash flow (None, perpetuity)
)

# find irr
irr = multiple_irr.find(initial_guess=0.05)
print(f"Internal Rate of Return (IRR) = {irr}")

# find modified irr
mirr = multiple_irr.find(initial_guess=0.05)
print(f"Internal Rate of Return (MIRR) = {mirr}")

# find all irr
irrs = multiple_irr.find_all([0.05,1])
print(f"All Internal Rate of Return (IRR) = {irrs}")

# check formula
formula_string = multiple_irr.formula_string
print(f"Formula = {formula_string}")

# get yield curve data
formula_string = multiple_irr.get_yield_curve(min_r=0, max_r=5, points=50)
print(f"Formula = {formula_string}")

# plot
fig, ax = multiple_irr.plot(
    min_r=0,
    max_r=5,
    points=100,
    figsize=(10,5),
    color='blue',
    )

# save image 
fig.savefig('irr.png')
